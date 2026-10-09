#include <algorithm>
#include <cmath>
#include <numeric>

#include "./occupation_sweep.hpp"

namespace triqs_cthyb {

  namespace {

    // C = A B into the storage of C: plain loops for the small blocks h_diag mostly has, BLAS for the large ones
    void multiply_into(matrix_t const &A, matrix_t const &B, matrix_t &C) {
      long const n = A.shape()[0], p = A.shape()[1], q = B.shape()[1];
      C.resize(n, q);
      if (n * p * q > 4096) {
        nda::blas::gemm(h_scalar_t{1}, A, B, h_scalar_t{0}, C);
        return;
      }
      for (long i = 0; i < n; ++i)
        for (long j = 0; j < q; ++j) {
          h_scalar_t x = 0;
          for (long l = 0; l < p; ++l) x += A(i, l) * B(l, j);
          C(i, j) = x;
        }
    }

    // op as a sum of parts that each map every block of h_diag to at most one block, with the same kinks(term) in all
    // their terms (get_op_mat refuses c^dagger_a c_b + c^dagger_b c_a, which sends a block to two blocks)
    template <typename Kinks> std::vector<std::pair<atom_diag::op_block_mat_t, int>> split(many_body_op_t const &op, atom_diag const &h_diag, Kinks kinks) {
      auto same_targets = [](auto const &x, auto const &y) {
        for (long b = 0; b < x.connection.size(); ++b)
          if (x.connection(b) != y.connection(b)) return false;
        return true;
      };
      std::vector<std::pair<atom_diag::op_block_mat_t, int>> parts;
      for (auto const &term : op) {
        auto mat      = h_diag.get_op_mat(many_body_op_t(term.coef, term.monomial));
        int const kin = kinks(term.monomial);
        auto same     = std::find_if(parts.begin(), parts.end(), [&](auto const &p) { return p.second == kin && same_targets(p.first, mat); });
        if (same == parts.end())
          parts.emplace_back(std::move(mat), kin);
        else
          for (long b = 0; b < mat.connection.size(); ++b)
            if (mat.connection(b) >= 0) same->first.block_mat[b] += mat.block_mat[b];
      }
      return parts;
    }

  } // namespace

  occupation_sweep::occupation_sweep(qmc_data const &data, triqs::mesh::dlr_imtime const &nodes, std::vector<many_body_op_t> const &ops_tau,
                                     std::vector<many_body_op_t> const &ops_0)
     : data(data), n_A(ops_tau.size()), n_B(ops_0.size()) {

    auto const &h_diag = data.h_diag;
    auto const &fops   = h_diag.get_fops();
    for (auto const *op_list : {&ops_tau, &ops_0})
      for (auto const &op : *op_list)
        for (auto const &term : op)
          if (term.monomial.size() % 2 != 0)
            TRIQS_RUNTIME_ERROR << "Imaginary-time correlators are measured for bosonic operators, an even number of c and c^dagger in "
                                   "every term. Got "
                                << op;

    // The occupation kinks of a term, as an index in kink_vectors: 0 for kinks the Lang-Firsov kernel does not see
    kink_vectors = {{}};
    auto kinks   = [&](triqs::operators::monomial_t const &monomial) {
      std::vector<double> dense(fops.size(), 0.0);
      for (auto const &c_op : monomial) dense[fops[c_op.indices]] += c_op.dagger ? 1.0 : -1.0;
      if (data.lang_firsov_blind_to(dense)) return 0;
      qmc_data::kinks_t sparse;
      for (long a = 0; a < long(dense.size()); ++a)
        if (dense[a] != 0.0) sparse.emplace_back(a, dense[a]);
      auto same = std::find(kink_vectors.begin(), kink_vectors.end(), sparse);
      if (same != kink_vectors.end()) return int(same - kink_vectors.begin());
      kink_vectors.push_back(sparse);
      return int(kink_vectors.size()) - 1;
    };

    // Kinks the kernel sees always take the trace to other blocks (the densities it couples are constant on every block),
    // and the Lang-Firsov weight of the configuration changes with them: those parts go across blocks
    within_blocks = true;
    auto add      = [&](std::vector<many_body_op_t> const &op_list, std::vector<part_t> &parts) {
      for (long i = 0; i < long(op_list.size()); ++i)
        for (auto &[mat, kin] : split(op_list[i], h_diag, kinks)) {
          for (long b = 0; b < mat.connection.size(); ++b) within_blocks = within_blocks && (mat.connection(b) < 0 || mat.connection(b) == b);
          within_blocks = within_blocks && kin == 0;
          parts.push_back({i, std::move(mat), kin});
        }
    };
    add(ops_tau, A_parts);
    add(ops_0, B_parts);

    // Within blocks, each operator in one piece, and the matrix elements (m, n) of every block where some A_i is nonzero
    int const n_blocks = h_diag.n_subspaces();
    if (within_blocks) {
      for (auto const &op : ops_tau) A_mat.push_back(h_diag.get_op_mat(op));
      for (auto const &op : ops_0) B_mat.push_back(h_diag.get_op_mat(op));
      terms.resize(n_blocks);
      for (int b = 0; b < n_blocks; ++b) {
        int const d = h_diag.get_subspace_dim(b);
        for (int m = 0; m < d; ++m)
          for (int n = 0; n < d; ++n) {
            bool active = false;
            for (auto const &A : A_mat) active = active || (A.connection(b) == b && std::abs(A.block_mat[b](m, n)) > 1e-13);
            if (active) terms[b].push_back({m, n});
          }
      }
    }

    // The nodes in increasing tau. Symmetrized, the k-th from the start and the k-th from the end are tau and beta - tau.
    long const n_nodes = nodes.size();
    double const beta  = data.config.beta();
    mesh_index.resize(n_nodes);
    std::iota(mesh_index.begin(), mesh_index.end(), 0);
    std::sort(mesh_index.begin(), mesh_index.end(), [&](long i, long j) { return nodes[i].value() < nodes[j].value(); });
    for (long i : mesh_index) node_tau.push_back(nodes[i].value());
    reflection.resize(n_nodes);
    for (long p = 0; p < n_nodes; ++p) {
      if (std::abs(node_tau[n_nodes - 1 - p] - (beta - node_tau[p])) > 1e-10 * beta)
        TRIQS_RUNTIME_ERROR << "occupation_sweep: the DLR nodes are not symmetric about beta / 2 (build them with symmetrize = true)";
      reflection[mesh_index[p]] = mesh_index[n_nodes - 1 - p];
    }

    values = nda::zeros<mc_weight_t>(n_nodes, n_A * n_B);
  }

  matrix_t const &occupation_sweep::op_matrix(int k, int b) const {
    auto const &op = ops[k].second;
    return op.dagger ? data.h_diag.cdag_matrix(op.linear_index, b) : data.h_diag.c_matrix(op.linear_index, b);
  }

  long occupation_sweep::op_target(int k, int b) const {
    auto const &op = ops[k].second;
    return op.dagger ? data.h_diag.cdag_connection(op.linear_index, b) : data.h_diag.c_connection(op.linear_index, b);
  }

  void occupation_sweep::accumulate(mc_weight_t s) {
    s *= data.atomic_reweighting;
    average_sign += s;

    // The trace operators in increasing time, those of the dynamical vertices included. Interval k is (t[k], t[k + 1])
    // with the nodes [first_node[k], first_node[k + 1]), and operator k sits at t[k + 1].
    ops = data.trace_ops();
    std::sort(ops.begin(), ops.end(), [](auto const &x, auto const &y) { return x.first < y.first; });
    int const n_ops = ops.size();
    t.resize(n_ops + 2);
    first_node.resize(n_ops + 2);
    t[0]                  = 0.0;
    t[n_ops + 1]          = data.config.beta();
    first_node[0]         = 0;
    first_node[n_ops + 1] = long(node_tau.size());
    for (int k = 0; k < n_ops; ++k) {
      t[k + 1]          = double(ops[k].first);
      first_node[k + 1] = std::lower_bound(node_tau.begin(), node_tau.end(), t[k + 1]) - node_tau.begin();
    }

    // Divided by the trace the Monte Carlo weight holds, checked against the full trace
    h_scalar_t const mc_trace       = data.atomic_weight * data.atomic_reweighting;
    auto const [bare_trace, trace_abs] = within_blocks ? accumulate_within_blocks(s / mc_trace) : accumulate_across_blocks(s / mc_trace);
    if (std::abs(bare_trace - mc_trace) > 1.e-8 * trace_abs)
      TRIQS_RUNTIME_ERROR << "Imaginary-time correlators: the trace " << bare_trace << " of configuration " << data.config.get_id()
                          << " differs from the Monte Carlo trace " << mc_trace;
  }

  std::pair<h_scalar_t, double> occupation_sweep::accumulate_within_blocks(mc_weight_t weight) {
    auto const &h_diag = data.h_diag;
    int const n_ops    = ops.size();
    h_scalar_t bare_trace = 0.0;
    double trace_abs      = 0.0;

    block.resize(n_ops + 1);
    decays.resize(n_ops + 1);
    R.resize(n_ops + 1);
    L.resize(n_ops + 1);
    Y.resize(n_B);
    coefficients.resize(n_A * n_B);

    for (int b0 = 0; b0 < h_diag.n_subspaces(); ++b0) {

      // The block in each interval, for the trace that starts in b0; it contributes only if it returns to b0
      block[0]    = b0;
      bool broken = false;
      for (int k = 0; k < n_ops && !broken; ++k) {
        block[k + 1] = op_target(k, block[k]);
        broken       = block[k + 1] < 0;
      }
      if (broken || block[n_ops] != b0) continue;

      // e^{-(t[k + 1] - t[k]) E} on the block of interval k
      for (int k = 0; k <= n_ops; ++k) {
        int const d = h_diag.get_subspace_dim(block[k]);
        decays[k].resize(d);
        for (int i = 0; i < d; ++i) decays[k][i] = std::exp(-(t[k + 1] - t[k]) * h_diag.get_eigenvalue(block[k], i));
      }

      int const d0 = h_diag.get_subspace_dim(b0);
      R[0]         = nda::eye<h_scalar_t>(d0);
      for (int k = 0; k < n_ops; ++k) {
        evolved.resize(R[k].shape());
        for (long i = 0; i < R[k].shape()[0]; ++i)
          for (long u = 0; u < R[k].shape()[1]; ++u) evolved(i, u) = decays[k][i] * R[k](i, u);
        multiply_into(op_matrix(k, block[k]), evolved, R[k + 1]);
      }
      L[n_ops] = nda::eye<h_scalar_t>(d0);
      for (int k = n_ops - 1; k >= 0; --k) {
        evolved.resize(L[k + 1].shape());
        for (long u = 0; u < L[k + 1].shape()[0]; ++u)
          for (long i = 0; i < L[k + 1].shape()[1]; ++i) evolved(u, i) = L[k + 1](u, i) * decays[k + 1][i];
        multiply_into(evolved, op_matrix(k, block[k]), L[k]);
      }

      h_scalar_t block_trace = 0.0;
      for (int u = 0; u < d0; ++u) block_trace += decays[n_ops][u] * R[n_ops](u, u);
      bare_trace += block_trace;
      trace_abs += std::abs(block_trace);

      bool any_B = false;
      for (auto const &B : B_mat) any_B = any_B || B.connection(b0) == b0;
      if (!any_B) continue;

      for (int k = 0; k <= n_ops; ++k) {
        long const p0 = first_node[k], p1 = first_node[k + 1];
        int const bk  = block[k];
        if (p0 == p1 || terms[bk].empty()) continue;

        for (long j = 0; j < n_B; ++j) {
          if (B_mat[j].connection(b0) != b0) continue;
          multiply_into(R[k], B_mat[j].block_mat[b0], RB);
          multiply_into(RB, L[k], Y[j]);
        }

        int const d = h_diag.get_subspace_dim(bk);
        left.resize(p1 - p0, d);
        right.resize(p1 - p0, d);
        for (long p = p0; p < p1; ++p)
          for (int i = 0; i < d; ++i) {
            double const E   = h_diag.get_eigenvalue(bk, i);
            left(p - p0, i)  = std::exp(-(t[k + 1] - node_tau[p]) * E);
            right(p - p0, i) = std::exp(-(node_tau[p] - t[k]) * E);
          }

        for (auto const &term : terms[bk]) {
          bool any = false;
          for (long i = 0; i < n_A; ++i) {
            h_scalar_t const a = (A_mat[i].connection(bk) == bk) ? A_mat[i].block_mat[bk](term.m, term.n) : h_scalar_t{0};
            for (long j = 0; j < n_B; ++j) {
              mc_weight_t const c = (a != h_scalar_t{0} && B_mat[j].connection(b0) == b0) ? weight * a * Y[j](term.n, term.m) : mc_weight_t{0};
              coefficients[i * n_B + j] = c;
              any                       = any || c != mc_weight_t{0};
            }
          }
          if (!any) continue;
          for (long p = p0; p < p1; ++p) {
            double const x = left(p - p0, term.m) * right(p - p0, term.n);
            for (long ij = 0; ij < n_A * n_B; ++ij) values(p, ij) += x * coefficients[ij];
          }
        }
      }
    }
    return {bare_trace, trace_abs};
  }

  // B_j takes the trace from block b at 0 to block y, the operators of the configuration take y to c at the node, A_i
  // takes c to d, and the operators after the node take d to b at beta. With R^(y)[k] the product from 0 to t_k that
  // starts in y and L^(d)[k] the one from t_{k+1} to beta that starts in d,
  //   <A_i(tau) B_j(0)> = sum_mn (A_i)_mn e^{-(t_{k+1} - tau) E_m} e^{-(tau - t_k) E_n} (R^(y)[k] B_j L^(d)[k])_nm.
  std::pair<h_scalar_t, double> occupation_sweep::accumulate_across_blocks(mc_weight_t weight) {
    auto const &h_diag = data.h_diag;
    int const n_ops    = ops.size();
    int const n_blocks = h_diag.n_subspaces();
    auto energy        = [&](long x, int i) { return h_diag.get_eigenvalue(x, i); };
    auto dim           = [&](long x) { return h_diag.get_subspace_dim(x); };

    // The intervals that hold a node, each with a slot for its products
    slot_of.assign(n_ops + 1, -1);
    int n_slots = 0, last_slot_k = -1;
    for (int k = 0; k <= n_ops; ++k)
      if (first_node[k] < first_node[k + 1]) {
        slot_of[k]  = n_slots++;
        last_slot_k = k;
      }
    R_at.resize(n_slots, std::vector<matrix_t>(n_blocks));
    L_at.resize(n_slots, std::vector<matrix_t>(n_blocks));
    R_block.assign(n_slots, std::vector<long>(n_blocks, -1));
    L_end.assign(n_slots, std::vector<long>(n_blocks, -1));

    // R^(y)[k] along the chain of blocks from y, up to the last interval with a node
    for (int y = 0; y < n_blocks; ++y) {
      matrix_t product = nda::eye<h_scalar_t>(dim(y));
      long x           = y;
      for (int k = 0; k <= last_slot_k; ++k) {
        if (slot_of[k] >= 0) {
          R_at[slot_of[k]][y]    = product;
          R_block[slot_of[k]][y] = x;
        }
        if (k == last_slot_k) break;
        evolved.resize(product.shape());
        for (long i = 0; i < product.shape()[0]; ++i)
          for (long u = 0; u < product.shape()[1]; ++u) evolved(i, u) = std::exp(-(t[k + 1] - t[k]) * energy(x, i)) * product(i, u);
        multiply_into(op_matrix(k, x), evolved, product);
        x = op_target(k, x);
        if (x < 0) break;
      }
    }

    // L^(z)[k] for every block z, from the last interval down: L^(z)[k - 1] = L^(z')[k] e^{-(t_{k+1} - t_k) H} O_{k-1}, with
    // z' the block O_{k-1} takes z to; end_level[z] is the block it ends in at beta
    L_level.resize(n_blocks);
    L_next.resize(n_blocks);
    end_level.resize(n_blocks);
    end_next.resize(n_blocks);
    for (int z = 0; z < n_blocks; ++z) {
      L_level[z]   = nda::eye<h_scalar_t>(dim(z));
      end_level[z] = z;
    }
    for (int k = n_ops;; --k) {
      if (slot_of[k] >= 0) {
        L_at[slot_of[k]]  = L_level;
        L_end[slot_of[k]] = end_level;
      }
      if (k == 0) break;
      for (int z = 0; z < n_blocks; ++z) {
        long const z1 = op_target(k - 1, z);
        end_next[z]   = (z1 < 0) ? -1 : end_level[z1];
        if (end_next[z] < 0) continue;
        evolved.resize(L_level[z1].shape());
        for (long u = 0; u < L_level[z1].shape()[0]; ++u)
          for (long i = 0; i < L_level[z1].shape()[1]; ++i) evolved(u, i) = L_level[z1](u, i) * std::exp(-(t[k + 1] - t[k]) * energy(z1, i));
        multiply_into(evolved, op_matrix(k - 1, z), L_next[z]);
      }
      std::swap(L_level, L_next);
      std::swap(end_level, end_next);
    }

    // The bare trace: from block x at 0 back to x at beta
    h_scalar_t bare_trace = 0.0;
    double trace_abs      = 0.0;
    for (int x = 0; x < n_blocks; ++x) {
      if (end_level[x] != x) continue;
      h_scalar_t block_trace = 0.0;
      for (int u = 0; u < dim(x); ++u) block_trace += L_level[x](u, u) * std::exp(-(t[1] - t[0]) * energy(x, u));
      bare_trace += block_trace;
      trace_abs += std::abs(block_trace);
    }

    // The Lang-Firsov potential of the kinks of A at every node and of those of B at 0, for the kinks the kernel sees
    long const n_kinks = kink_vectors.size();
    phi_A.resize(long(node_tau.size()), n_kinks);
    phi_B.assign(n_kinks, 0.0);
    phi_A() = 0.0;
    for (long kin = 1; kin < n_kinks; ++kin) {
      for (long p = 0; p < long(node_tau.size()); ++p) phi_A(p, kin) = data.lang_firsov_potential(kink_vectors[kin], node_tau[p]);
      phi_B[kin] = data.lang_firsov_potential(kink_vectors[kin], 0.0);
    }

    std::vector<double> right_c;
    for (int k = 0; k <= last_slot_k; ++k) {
      int const slot = slot_of[k];
      if (slot < 0) continue;
      for (auto const &B : B_parts)
        for (int b = 0; b < n_blocks; ++b) {
          long const y = B.mat.connection(b);
          if (y < 0) continue;
          long const c = R_block[slot][y];
          if (c < 0) continue;
          multiply_into(R_at[slot][y], B.mat.block_mat[b], RB); // from b to c
          for (auto const &A : A_parts) {
            long const d = A.mat.connection(c);
            if (d < 0 || L_end[slot][d] != b) continue;
            multiply_into(RB, L_at[slot][d], Y_part); // from d to c, round the trace
            auto const &a = A.mat.block_mat[c];       // from c to d
            for (long p = first_node[k]; p < first_node[k + 1]; ++p) {
              right_c.resize(dim(c));
              for (int n = 0; n < dim(c); ++n) right_c[n] = std::exp(-(node_tau[p] - t[k]) * energy(c, n));
              mc_weight_t x = 0;
              for (int m = 0; m < dim(d); ++m) {
                double const left_m = std::exp(-(t[k + 1] - node_tau[p]) * energy(d, m));
                for (int n = 0; n < dim(c); ++n) x += a(m, n) * left_m * right_c[n] * Y_part(n, m);
              }
              if (A.kinks != 0 || B.kinks != 0)
                x *= std::exp(phi_A(p, A.kinks) + phi_B[B.kinks]
                              + data.lang_firsov_interaction(kink_vectors[A.kinks], kink_vectors[B.kinks], node_tau[p]));
              values(p, A.op * n_B + B.op) += weight * x;
            }
          }
        }
    }
    return {bare_trace, trace_abs};
  }

  nda::array<mc_weight_t, 3> occupation_sweep::collect(mpi::communicator const &c) {
    average_sign = mpi::all_reduce(average_sign, c);
    values       = mpi::all_reduce(values, c);

    double const norm = std::real(average_sign);
    nda::array<mc_weight_t, 3> result(long(node_tau.size()), n_A, n_B);
    for (long p = 0; p < long(node_tau.size()); ++p)
      for (long i = 0; i < n_A; ++i)
        for (long j = 0; j < n_B; ++j) result(mesh_index[p], i, j) = values(p, i * n_B + j) / norm;
    return result;
  }

} // namespace triqs_cthyb
