#include <algorithm>
#include <array>
#include <cmath>

#include "./occupation_sweep.hpp"

namespace triqs_cthyb {

  namespace {

    // Every term a product of number operators. The blocks of h_diag are spanned by Fock states, so such an operator maps
    // each block to itself: the trace with it runs through the same blocks as the bare one, and it leaves the occupation
    // kinks, hence the Lang-Firsov weight, alone. Two of them commute, which gives O(0) = O(beta).
    bool is_occupation_diagonal(many_body_op_t const &op) {
      for (auto const &[monomial, coeff] : op) {
        std::vector<triqs::operators::indices_t> created, annihilated;
        for (auto const &c_op : monomial) (c_op.dagger ? created : annihilated).push_back(c_op.indices);
        std::sort(created.begin(), created.end());
        std::sort(annihilated.begin(), annihilated.end());
        if (created != annihilated) return false;
      }
      return true;
    }

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

  } // namespace

  occupation_sweep::occupation_sweep(qmc_data const &data, long n_tau, std::vector<many_body_op_t> const &ops_tau,
                                     std::vector<many_body_op_t> const &ops_0)
     : data(data), n_tau(n_tau), dtau(data.config.beta() / double(n_tau - 1)), n_A(ops_tau.size()), n_B(ops_0.size()) {

    for (auto const *ops : {&ops_tau, &ops_0})
      for (auto const &op : *ops)
        if (!is_occupation_diagonal(op))
          TRIQS_RUNTIME_ERROR << "Imaginary-time correlators are measured for operators diagonal in the occupation basis only, every "
                                 "term a product of number operators (n_a, N, S_z, n_a n_b, ...), so that the trace with them runs "
                                 "through the same blocks of h_loc as the bare one. Got "
                              << op;

    auto const &h_diag = data.h_diag;
    for (auto const &op : ops_tau) A_mat.push_back(h_diag.get_op_mat(op));
    for (auto const &op : ops_0) B_mat.push_back(h_diag.get_op_mat(op));

    // The terms (m, n) of every block where some A_i is nonzero, by E_m - E_n. A difference below RATE_TOL is a
    // degeneracy, and rates closer than RATE_TOL are one rate: either is off by at most beta * RATE_TOL relative.
    double const RATE_TOL = 1e-10;
    int const n_blocks    = h_diag.n_subspaces();
    terms.resize(n_blocks);
    std::vector<double> rates;
    for (int b = 0; b < n_blocks; ++b) {
      int const d = h_diag.get_subspace_dim(b);
      for (int m = 0; m < d; ++m)
        for (int n = 0; n < d; ++n) {
          bool active = false;
          for (auto const &A : A_mat) active = active || (A.connection(b) == b && std::abs(A.block_mat[b](m, n)) > 1e-13);
          if (!active) continue;
          double const omega  = h_diag.get_eigenvalue(b, m) - h_diag.get_eigenvalue(b, n);
          bool const constant = std::abs(omega) < RATE_TOL;
          terms[b].push_back({m, n, constant ? 0 : -1, constant ? 0.0 : std::abs(omega), omega > 0});
          if (!constant) rates.push_back(std::abs(omega));
        }
    }
    std::sort(rates.begin(), rates.end());
    std::vector<double> distinct; // the smallest rate of each group
    for (double r : rates)
      if (distinct.empty() || r - distinct.back() > RATE_TOL) distinct.push_back(r);

    // An accumulator per rate and direction while they fit in MAX_ACCUMULATED elements, the rest evaluated directly
    long const n_pairs = n_A * n_B;
    slots              = {{1.0, false}};
    std::vector<std::array<int, 2>> slot_of(distinct.size(), {-2, -2}); // by rate and backward, -2 for not yet assigned
    bool any_direct = false;
    for (auto &block_terms : terms)
      for (auto &term : block_terms) {
        if (term.slot == 0) continue;
        long const q = std::upper_bound(distinct.begin(), distinct.end(), term.rate) - distinct.begin() - 1;
        term.rate    = distinct[q];
        int &slot    = slot_of[q][term.backward];
        if (slot == -2) {
          if (long(slots.size() + 1) * n_tau * n_pairs <= MAX_ACCUMULATED) {
            slot = slots.size();
            slots.push_back({std::exp(-term.rate * dtau), term.backward});
          } else
            slot = -1;
        }
        term.slot  = slot;
        any_direct = any_direct || slot == -1;
      }

    accumulated = nda::zeros<mc_weight_t>(long(slots.size()), n_tau, n_pairs);
    if (any_direct) direct = nda::zeros<mc_weight_t>(n_tau, n_pairs);
  }

  void occupation_sweep::accumulate(mc_weight_t s) {
    s *= data.atomic_reweighting;
    average_sign += s;

    auto const &h_diag = data.h_diag;
    double const beta  = data.config.beta();

    // The trace operators in increasing time, those of the dynamical vertices included. Interval k is (t[k], t[k + 1])
    // with the mesh points [first_point[k], first_point[k + 1]), and operator k sits at t[k + 1].
    auto ops = data.trace_ops();
    std::sort(ops.begin(), ops.end(), [](auto const &x, auto const &y) { return x.first < y.first; });
    int const n_ops = ops.size();
    t.resize(n_ops + 2);
    first_point.resize(n_ops + 2);
    t[0]                   = 0.0;
    t[n_ops + 1]           = beta;
    first_point[0]         = 0;
    first_point[n_ops + 1] = n_tau;
    for (int k = 0; k < n_ops; ++k) {
      t[k + 1] = double(ops[k].first);
      long p   = std::min(n_tau, long(std::ceil(t[k + 1] / dtau)));
      while (p > 0 && double(p - 1) * dtau >= t[k + 1]) --p;
      while (p < n_tau && double(p) * dtau < t[k + 1]) ++p;
      first_point[k + 1] = p;
    }

    auto op_target = [&](int k, int b) -> long {
      auto const &op = ops[k].second;
      return op.dagger ? h_diag.cdag_connection(op.linear_index, b) : h_diag.c_connection(op.linear_index, b);
    };
    auto op_matrix = [&](int k, int b) -> matrix_t const & {
      auto const &op = ops[k].second;
      return op.dagger ? h_diag.cdag_matrix(op.linear_index, b) : h_diag.c_matrix(op.linear_index, b);
    };

    // Divided by the trace the Monte Carlo weight holds, checked against the full trace below
    h_scalar_t const mc_trace = data.atomic_weight * data.atomic_reweighting;
    mc_weight_t const weight  = s / mc_trace;
    h_scalar_t bare_trace     = 0.0;
    double trace_abs          = 0.0;

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
        long const p0 = first_point[k], p1 = first_point[k + 1];
        int const bk  = block[k];
        if (p0 == p1 || terms[bk].empty()) continue;

        for (long j = 0; j < n_B; ++j) {
          if (B_mat[j].connection(b0) != b0) continue;
          multiply_into(R[k], B_mat[j].block_mat[b0], RB);
          multiply_into(RB, L[k], Y[j]);
        }

        for (auto const &term : terms[bk]) {
          bool any = false;
          for (long i = 0; i < n_A; ++i) {
            h_scalar_t const a = (A_mat[i].connection(bk) == bk) ? A_mat[i].block_mat[bk](term.m, term.n) : h_scalar_t{0};
            for (long j = 0; j < n_B; ++j) {
              mc_weight_t c = 0;
              if (a != h_scalar_t{0} && B_mat[j].connection(b0) == b0) c = weight * a * Y[j](term.n, term.m);
              coefficients[i * n_B + j] = c;
              any                       = any || c != mc_weight_t{0};
            }
          }
          if (any)
            deposit(term, p0, p1, t[k], t[k + 1], h_diag.get_eigenvalue(bk, term.m), h_diag.get_eigenvalue(bk, term.n));
        }
      }
    }

    if (std::abs(bare_trace - mc_trace) > 1.e-8 * trace_abs)
      TRIQS_RUNTIME_ERROR << "Imaginary-time correlators: the trace " << bare_trace << " of configuration " << data.config.get_id()
                          << " differs from the Monte Carlo trace " << mc_trace;
  }

  // coefficients * e^{-(t_right - tau) E_m} e^{-(tau - t_left) E_n} on the mesh points [p0, p1) of the interval (t_left, t_right)
  void occupation_sweep::deposit(term_t const &term, long p0, long p1, double t_left, double t_right, double E_m, double E_n) {
    long const n_pairs = n_A * n_B;
    auto add           = [&](mc_weight_t *row, double x) {
      for (long ij = 0; ij < n_pairs; ++ij) row[ij] += x * coefficients[ij];
    };
    double const length = t_right - t_left;

    // Constant: added at p0, taken off again at p1
    if (term.slot == 0) {
      double const x = std::exp(-length * E_m);
      add(&accumulated(0, p0, 0), x);
      if (p1 < n_tau) add(&accumulated(0, p1, 0), -x);
      return;
    }

    // K e^{-rate (t_right - tau)} backward, K e^{-rate (tau - t_left)} forward
    double const K = std::exp(-length * (term.backward ? E_n : E_m));
    auto distance  = [&](long p) { return term.backward ? t_right - double(p) * dtau : double(p) * dtau - t_left; };
    long const enter = term.backward ? p1 - 1 : p0; // the first point of the interval in the direction of decay
    long const leave = term.backward ? p0 - 1 : p1; // the first point past it

    if (term.slot > 0) {
      add(&accumulated(term.slot, enter, 0), K * std::exp(-term.rate * distance(enter)));
      if (leave >= 0 && leave < n_tau) add(&accumulated(term.slot, leave, 0), -K * std::exp(-term.rate * distance(leave)));
      return;
    }

    double const step = std::exp(-term.rate * dtau);
    double x          = K * std::exp(-term.rate * distance(enter));
    for (long p = enter; p != leave; p += (term.backward ? -1 : 1), x *= step) add(&direct(p, 0), x);
  }

  nda::array<mc_weight_t, 3> occupation_sweep::collect(mpi::communicator const &c) {
    average_sign = mpi::all_reduce(average_sign, c);
    accumulated  = mpi::all_reduce(accumulated, c);
    if (direct.size() > 0) direct = mpi::all_reduce(direct, c);

    // Each accumulator through its recursion, decaying along its direction
    long const n_pairs = n_A * n_B;
    nda::array<mc_weight_t, 2> sum(n_tau, n_pairs);
    if (direct.size() > 0)
      sum = direct;
    else
      sum = 0;
    std::vector<mc_weight_t> running(n_pairs);
    for (long q = 0; q < long(slots.size()); ++q) {
      std::fill(running.begin(), running.end(), mc_weight_t{0});
      for (long step = 0; step < n_tau; ++step) {
        long const p = slots[q].backward ? n_tau - 1 - step : step;
        for (long ij = 0; ij < n_pairs; ++ij) {
          running[ij] = slots[q].decay * running[ij] + accumulated(q, p, ij);
          sum(p, ij) += running[ij];
        }
      }
    }

    nda::array<mc_weight_t, 3> result(n_tau, n_A, n_B);
    double const norm = std::real(average_sign);
    for (long p = 0; p < n_tau; ++p)
      for (long i = 0; i < n_A; ++i)
        for (long j = 0; j < n_B; ++j) result(p, i, j) = sum(p, i * n_B + j) / norm;
    return result;
  }

} // namespace triqs_cthyb
