/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2014, P. Seth, I. Krivenko, M. Ferrero and O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#include "./dynamical_interactions.hpp"
#include "./math_utils.hpp"
#include <triqs/utility/exceptions.hpp>
#include <set>

namespace triqs_cthyb {

  using namespace triqs::operators; // for c<h_scalar_t>(...), c_dag<h_scalar_t>(...)

  namespace {

    // (block_index, inner_index) -> is there any non-zero data anywhere in this
    // scalar-valued slice's mesh?
    bool any_nonzero(gf_const_view<imtime, matrix_valued> block, int i1, int i2, double threshold = 1.e-13) {
      for (auto const &tau_pt : block.mesh())
        if (std::abs(block[tau_pt](i1, i2)) > threshold) return true;
      return false;
    }

    bool any_nonzero(gf_const_view<imtime, matrix_valued> block, double threshold = 1.e-13) {
      return max_element(nda::abs(block.data())) > threshold;
    }

    // Extract a single (i1, i2) scalar component of a matrix-valued block as its own
    // scalar-valued gf, dividing by `factor` -- the shared last step of both D0 and
    // Jperp expansion.
    gf<imtime, scalar_valued> scalar_component(gf_const_view<imtime, matrix_valued> block, int i1, int i2, double factor = 1.0) {
      auto coupling = gf<imtime, scalar_valued>{block.mesh()};
      for (auto const &tau_pt : block.mesh()) coupling[tau_pt] = real(block[tau_pt](i1, i2)) * factor;
      return coupling;
    }

    double eval_scalar_gf(gf<imtime, scalar_valued> const &g, double tau) { return real(g[closest_mesh_pt(tau)]); }

    // Are two couplings numerically the same curve? Used to detect a "sufficiently
    // symmetric" set of vertices that all share one physical coupling (see
    // find_total_density_decomposition).
    bool gf_close(gf<imtime, scalar_valued> const &g1, gf<imtime, scalar_valued> const &g2, double threshold = 1.e-10) {
      if (g1.mesh().size() != g2.mesh().size()) return false;
      for (auto const &tau_pt : g1.mesh())
        if (std::abs(g1[tau_pt] - g2[tau_pt]) > threshold) return false;
      return true;
    }

    int count_orbitals(std::map<std::pair<int, int>, int> const &linindex) {
      int n = 0;
      for (auto const &pair : linindex) n = std::max(n, pair.second + 1);
      return n;
    }

    // Bare (pre-shift) density-density matrix reading of h_loc, matching the
    // Kanamori-style U_matrix(i,j) n_i n_j (i!=j) / mu_vec(i) n_i term shapes -- purely
    // for the diagnostic printout in apply_lang_firsov_shift below (as in CTSEG).
    // classify_dyn_vertices / the shift math never depend on h_loc having this shape;
    // terms that don't match it (e.g. hopping, spin-flip) simply don't contribute here.
    std::pair<nda::matrix<double>, nda::vector<double>> bare_density_matrix(many_body_op_t const &h_loc, fundamental_operator_set const &fops,
                                                                            int n_orbitals) {
      nda::matrix<double> U_matrix(n_orbitals, n_orbitals);
      nda::vector<double> mu_vec(n_orbitals);
      U_matrix = 0.0;
      mu_vec   = 0.0;
      for (auto const &[term, coeff] : h_loc) {
        if (term.size() == 2) {
          if (term[0].dagger && !term[1].dagger && term[0].indices == term[1].indices) mu_vec(fops[term[0].indices]) -= real(coeff);
        } else if (term.size() == 4) {
          if (term[0].dagger && term[1].dagger && !term[2].dagger && !term[3].dagger && term[0].indices == term[3].indices
              && term[1].indices == term[2].indices) {
            int i = fops[term[0].indices];
            int j = fops[term[1].indices];
            if (i != j) {
              U_matrix(i, j) += real(coeff);
              U_matrix(j, i) += real(coeff);
            }
          }
        }
      }
      return {U_matrix, mu_vec};
    }

  } // namespace

  // -----------------------------------------------------------------------------------

  op_desc_pair_t extract_bilinear(many_body_op_t const &op, fundamental_operator_set const &fops,
                                  std::map<std::pair<int, int>, int> const &linindex, std::string const &op_name) {

    int n_terms = 0;
    monomial_t the_monomial;
    for (auto const &[monomial, coeff] : op) {
      ++n_terms;
      the_monomial = monomial;
    }
    if (n_terms != 1)
      TRIQS_RUNTIME_ERROR << op_name << " must be a single fermion bilinear (e.g. c_dag('up',0)*c('down',0)), but has " << n_terms
                          << " terms.";
    if (the_monomial.size() != 2)
      TRIQS_RUNTIME_ERROR << op_name << " must be a bilinear (exactly one creation and one annihilation operator), but has "
                          << the_monomial.size() << " operators.";
    if (the_monomial[0].dagger == the_monomial[1].dagger)
      TRIQS_RUNTIME_ERROR << op_name << " must contain one creation and one annihilation operator, but has two "
                          << (the_monomial[0].dagger ? "creation" : "annihilation") << " operators.";

    // Reverse-lookup (block_index, inner_index) from the linear index fops already
    // assigns each fundamental operator -- exactly the same primitive the existing
    // h_loc term-matcher used (fops[indices] gives the linear index directly).
    auto to_op_desc = [&](auto const &fermion_op) -> op_desc {
      int lin = fops[fermion_op.indices];
      for (auto const &[block_inner, linear] : linindex)
        if (linear == lin) return op_desc{block_inner.first, block_inner.second, fermion_op.dagger, lin};
      TRIQS_RUNTIME_ERROR << "extract_bilinear: linear index " << lin << " (from " << op_name << ") not found in linindex; "
                          << "is this operator built from a block/orbital outside gf_struct?";
    };

    auto const &dag_op    = the_monomial[0].dagger ? the_monomial[0] : the_monomial[1];
    auto const &nondag_op = the_monomial[0].dagger ? the_monomial[1] : the_monomial[0];
    return {to_op_desc(dag_op), to_op_desc(nondag_op)};
  }

  // -----------------------------------------------------------------------------------

  bool is_density_bilinear(op_desc_pair_t const &bp) {
    return bp.opL.block_index == bp.opR.block_index && bp.opL.inner_index == bp.opR.inner_index;
  }

  // -----------------------------------------------------------------------------------

  void expand_D0_into_vertices(block2_gf_const_view<imtime> D0t, gf_struct_t const &gf_struct, std::vector<dyn_vertex_t> &vertices) {
    // D0(tau) n_a(tau) n_b(0): unlike Jperp, a density-density coupling needs no
    // spin/orbital convention -- every non-zero (bl1, i1, bl2, i2) entry is its own
    // vertex, whatever bl1 and bl2 are.
    for (size_t bl1 = 0; bl1 < gf_struct.size(); ++bl1) {
      for (size_t bl2 = 0; bl2 < gf_struct.size(); ++bl2) {
        auto D0_bl = D0t(bl1, bl2);
        if (!any_nonzero(D0_bl)) continue;

        auto bl1_name = gf_struct[bl1].first;
        auto bl2_name = gf_struct[bl2].first;
        int bl1_size  = gf_struct[bl1].second;
        int bl2_size  = gf_struct[bl2].second;

        for (int i1 = 0; i1 < bl1_size; ++i1) {
          for (int i2 = 0; i2 < bl2_size; ++i2) {
            if (!any_nonzero(D0_bl, i1, i2)) continue;
            vertices.push_back({c_dag<h_scalar_t>(bl1_name, i1) * c<h_scalar_t>(bl1_name, i1),
                                c_dag<h_scalar_t>(bl2_name, i2) * c<h_scalar_t>(bl2_name, i2), scalar_component(D0_bl, i1, i2)});
          }
        }
      }
    }
  }

  // -----------------------------------------------------------------------------------

  void expand_Jperp_into_vertices(gf_const_view<imtime, matrix_valued> Jperpt, gf_struct_t const &gf_struct, std::vector<dyn_vertex_t> &vertices) {
    if (!any_nonzero(Jperpt)) return;

    // Jperp_tau is a single global up/down coupling with no orbital index of its own
    // (matches ctseg exactly), so it only applies to the unambiguous case of exactly 2
    // blocks, each a single fermion mode. For per-orbital-pair or inter-orbital
    // spin-flip, use solver.add_dyn_vertex(...) directly -- nothing here is inferred.
    if (gf_struct.size() != 2)
      TRIQS_RUNTIME_ERROR << "Jperp_tau (spin-flip) is a single global coupling and only supports exactly 2 blocks "
                             "(e.g. spin up/down), matching ctseg. Found "
                          << gf_struct.size()
                          << " blocks. For per-orbital-pair or inter-orbital spin-flip, use "
                             "solver.add_dyn_vertex(...) directly, specifying each vertex's operators explicitly.";
    if (gf_struct[0].second != 1 || gf_struct[1].second != 1)
      TRIQS_RUNTIME_ERROR << "Jperp_tau (spin-flip) is a single global coupling with no orbital index, so both blocks "
                             "must have exactly 1 orbital; got sizes "
                          << gf_struct[0].second << " and " << gf_struct[1].second
                          << ". For multi-orbital spin-flip, use solver.add_dyn_vertex(...) directly.";

    auto bl0_name = gf_struct[0].first;
    auto bl1_name = gf_struct[1].first;
    auto coupling = scalar_component(Jperpt, 0, 0, 0.5); // Jperp(tau)/2, as in the original derivation

    // S+(tau) S-(0): c_dag(bl0,0) c(bl1,0) (tau) * c_dag(bl1,0) c(bl0,0) (0)
    vertices.push_back(
       {c_dag<h_scalar_t>(bl0_name, 0) * c<h_scalar_t>(bl1_name, 0), c_dag<h_scalar_t>(bl1_name, 0) * c<h_scalar_t>(bl0_name, 0), coupling});
    // S-(tau) S+(0): c_dag(bl1,0) c(bl0,0) (tau) * c_dag(bl0,0) c(bl1,0) (0)
    vertices.push_back(
       {c_dag<h_scalar_t>(bl1_name, 0) * c<h_scalar_t>(bl0_name, 0), c_dag<h_scalar_t>(bl0_name, 0) * c<h_scalar_t>(bl1_name, 0), coupling});
  }

  // -----------------------------------------------------------------------------------

  std::vector<dyn_vertex_t> collect_dyn_vertices(std::vector<dyn_vertex_t> const &explicit_vertices, block2_gf_const_view<imtime> D0t,
                                                 gf_const_view<imtime, matrix_valued> Jperpt, gf_struct_t const &gf_struct) {
    std::vector<dyn_vertex_t> vertices = explicit_vertices;
    expand_D0_into_vertices(D0t, gf_struct, vertices);
    expand_Jperp_into_vertices(Jperpt, gf_struct, vertices);
    return vertices;
  }

  // -----------------------------------------------------------------------------------

  classified_dyn_vertices_t classify_dyn_vertices(std::vector<dyn_vertex_t> const &vertices, many_body_op_t const &h_loc,
                                                  fundamental_operator_set const &fops, std::map<std::pair<int, int>, int> const &linindex,
                                                  bool lang_firsov_requested) {
    auto commutes_with_hloc = [&](many_body_op_t const &op) { return (op * h_loc - h_loc * op).is_almost_zero(); };

    classified_dyn_vertices_t result;
    for (auto const &v : vertices) {
      bool eligible = false;
      if (lang_firsov_requested) {
        auto bp1 = extract_bilinear(v.op1, fops, linindex, "op1");
        auto bp2 = extract_bilinear(v.op2, fops, linindex, "op2");
        eligible = is_density_bilinear(bp1) && is_density_bilinear(bp2) && commutes_with_hloc(v.op1) && commutes_with_hloc(v.op2);
      }
      (eligible ? result.lang_firsov : result.stochastic).push_back(v);
    }
    return result;
  }

  // -----------------------------------------------------------------------------------

  void apply_lang_firsov_shift(many_body_op_t &h_loc, std::vector<dyn_vertex_t> const &lf_vertices, fundamental_operator_set const &fops,
                               std::map<std::pair<int, int>, int> const &linindex, double beta, int N_leg, int verbosity) {
    if (lf_vertices.empty()) return;

    // Aggregate before/after view of the shift, as a density-density matrix over all
    // orbitals (as in CTSEG) -- independent of how many orbitals/blocks are in play,
    // and of whether the eligible vertices came from D0_tau or explicit add_dyn_vertex
    // density couplings.
    int n_orb                = count_orbitals(linindex);
    auto [U_matrix, mu_vec]  = bare_density_matrix(h_loc, fops, n_orb);
    nda::matrix<double> U_renorm  = U_matrix;
    nda::vector<double> mu_renorm = mu_vec;

    if (verbosity >= 2) {
      std::cout << "\n Interaction matrix: U =" << std::endl << U_matrix << std::endl;
      std::cout << "\nOrbital energies: mu - eps = " << mu_vec << std::endl;
    }

    for (auto const &v : lf_vertices) {
      auto bp1 = extract_bilinear(v.op1, fops, linindex, "op1"); // guaranteed a density bilinear: classify_dyn_vertices checked
      auto bp2 = extract_bilinear(v.op2, fops, linindex, "op2");

      int n_pt_tau = v.coupling.mesh().size();
      auto d_n     = fit_legendre_coeffs(n_pt_tau, beta, [&v](double tau) { return eval_scalar_gf(v.coupling, tau); }, N_leg);
      double d0       = d_n(0);
      double d1       = (N_leg > 1) ? d_n(1) : 0.0;
      double Kprime_0 = -1.0 * beta * (d0 - d1 / 3.0);
      if (std::abs(Kprime_0) < 1.e-13) continue;

      int lin1 = bp1.opL.linear_index;
      int lin2 = bp2.opL.linear_index;

      if (lin1 == lin2) {
        // Diagonal: chemical-potential shift H -> H - 0.5 * K'(0) * n
        h_loc = h_loc - 0.5 * Kprime_0 * v.op1;
        mu_renorm(lin1) += 0.5 * Kprime_0;
      } else {
        // Off-diagonal: H -> H - 0.5 * K'(0) * n_1 * n_2. Vertices for both orbital
        // orderings (a,b) and (b,a) are expected in lf_vertices (that's how
        // expand_D0_into_vertices enumerates them), giving the total 1.0*K'(0) factor;
        // the symmetric double-write below mirrors that same convention in U_renorm,
        // and stays correct even for a single explicit (unpaired) add_dyn_vertex too.
        h_loc = h_loc - 0.5 * Kprime_0 * v.op1 * v.op2;
        U_renorm(lin1, lin2) -= 0.5 * Kprime_0;
        U_renorm(lin2, lin1) -= 0.5 * Kprime_0;
      }

      if (verbosity >= 2)
        std::cout << "Lang-Firsov K'(0) shift: K'(0)=" << Kprime_0 << " for vertex " << v.op1 << " -- " << v.op2 << std::endl;
    }

    if (verbosity >= 2) {
      std::cout << "\n Renormalized interaction matrix: U =" << std::endl << U_renorm << std::endl;
      std::cout << "\nRenormalized orbital energies: mu - eps = " << mu_renorm << std::endl;
    }
  }

  // -----------------------------------------------------------------------------------

  std::vector<std::vector<std::vector<double>>> build_K_n(std::vector<dyn_vertex_t> const &lf_vertices, double beta,
                                                          std::map<std::pair<int, int>, int> const &linindex,
                                                          fundamental_operator_set const &fops, int N_leg) {
    std::vector<std::vector<std::vector<double>>> K_n;
    if (lf_vertices.empty()) return K_n;

    auto M_matrix = build_M_matrix(N_leg, beta);
    int max_linindex = 0;
    for (auto const &pair : linindex) max_linindex = std::max(max_linindex, pair.second);
    K_n.resize(max_linindex + 1, std::vector<std::vector<double>>(max_linindex + 1, std::vector<double>(N_leg, 0.0)));

    for (auto const &v : lf_vertices) {
      auto bp1 = extract_bilinear(v.op1, fops, linindex, "op1");
      auto bp2 = extract_bilinear(v.op2, fops, linindex, "op2");
      int n_pt_tau = v.coupling.mesh().size();
      auto d_n     = fit_legendre_coeffs(n_pt_tau, beta, [&v](double tau) { return eval_scalar_gf(v.coupling, tau); }, N_leg);
      nda::vector<double> k_n_vec = M_matrix * d_n;

      int lin1 = bp1.opL.linear_index;
      int lin2 = bp2.opL.linear_index;
      for (int n = 0; n < N_leg; ++n) K_n[lin1][lin2][n] = k_n_vec(n);
    }
    return K_n;
  }

  // -----------------------------------------------------------------------------------

  void fold_into_stochastic_catalog(std::vector<dyn_vertex_t> const &stoch_vertices, fundamental_operator_set const &fops,
                                    std::map<std::pair<int, int>, int> const &linindex, std::vector<bosonic_op_pair_t> &dyn_op_list,
                                    std::vector<std::function<double(double)>> &dyn_interactions) {
    for (auto const &v : stoch_vertices) {
      auto bp1 = extract_bilinear(v.op1, fops, linindex, "op1");
      auto bp2 = extract_bilinear(v.op2, fops, linindex, "op2");

      int f_index = static_cast<int>(dyn_interactions.size());
      auto coupling_copy = v.coupling; // copy for lambda capture, mirrors the existing D0/Jperp lambda pattern
      dyn_interactions.emplace_back([coupling_copy](double tau) -> double { return eval_scalar_gf(coupling_copy, tau); });
      dyn_op_list.push_back({bp1, bp2, f_index});
    }
  }

  // -----------------------------------------------------------------------------------

  std::vector<nda::vector<double>> find_conserved_density_combinations(many_body_op_t const &h_loc, fundamental_operator_set const &fops,
                                                                        std::map<std::pair<int, int>, int> const &linindex) {
    int M = count_orbitals(linindex);
    if (M == 0) return {};

    // [n_a, h_loc] for every orbital a, indexed by linear index. n_a must be built from
    // fops's own indices_t for that linear position (via the data_t conversion) -- NOT
    // from linindex's (block_index, inner_index) key directly, which uses gf_struct's
    // block *position*, not its name, and would silently build an operator on a
    // fundamental mode unrelated to h_loc's actual operator algebra (making every
    // commutator trivially, and wrongly, zero).
    auto fops_indices = fundamental_operator_set::data_t(fops);
    std::vector<many_body_op_t> commutators(M);
    for (auto const &[block_inner, a] : linindex) {
      auto const &indices_a = fops_indices[a];
      auto c_dag_a           = many_body_op_t::make_canonical(true, indices_a);
      auto c_a               = many_body_op_t::make_canonical(false, indices_a);
      auto n_a               = c_dag_a * c_a;
      commutators[a]         = n_a * h_loc - h_loc * n_a;
    }

    // Collect every distinct monomial appearing in any commutator, as rows of a real
    // matrix -- real and imaginary parts of each coefficient kept as separate rows, so
    // this is correct whether h_scalar_t is real or complex.
    std::map<monomial_t, nda::vector<double>> real_rows, imag_rows;
    auto accumulate = [&](std::map<monomial_t, nda::vector<double>> &rows, monomial_t const &monomial, int a, double value) {
      if (std::abs(value) < 1.e-13) return;
      auto it = rows.find(monomial);
      if (it == rows.end()) {
        nda::vector<double> v(M);
        v  = 0.0;
        it = rows.emplace(monomial, v).first;
      }
      it->second(a) += value;
    };
    for (int a = 0; a < M; ++a)
      for (auto const &[monomial, coeff] : commutators[a]) {
        accumulate(real_rows, monomial, a, real(coeff));
        accumulate(imag_rows, monomial, a, imag(coeff));
      }

    int P = static_cast<int>(real_rows.size() + imag_rows.size());
    if (P == 0) return {}; // every n_a already commutes with h_loc individually -- classify_dyn_vertices handles this alone.

    nda::matrix<double> mat(P, M);
    mat     = 0.0;
    int row = 0;
    for (auto const &[monomial, vec] : real_rows) { mat(row, nda::range::all) = vec; ++row; }
    for (auto const &[monomial, vec] : imag_rows) { mat(row, nda::range::all) = vec; ++row; }

    auto [U, s, Vt]  = nda::linalg::svd(mat);
    double s_max     = (s.size() > 0) ? s(0) : 0.0;
    double threshold = 1.e-9 * std::max(s_max, 1.0);

    std::vector<nda::vector<double>> conserved;
    for (int i = 0; i < M; ++i) {
      bool is_null = (i >= s.size()) || (s(i) < threshold);
      if (!is_null) continue;
      nda::vector<double> v(M);
      for (int k = 0; k < M; ++k) v(k) = Vt(i, k);
      conserved.push_back(v);
    }
    return conserved;
  }

  // -----------------------------------------------------------------------------------

  namespace {

    // Least-squares fit of target (n x n) as sum_k gamma_k * design[k], returning the
    // achieved max-abs residual. SVD-based (pseudo-inverse), robust to a rank-deficient
    // design -- the design matrices (outer products of conserved combinations) need not
    // be linearly independent even when the combinations themselves are.
    double fit_and_residual(std::vector<nda::matrix<double>> const &design, nda::matrix<double> const &target, int n,
                            nda::vector<double> &gamma) {
      int K = static_cast<int>(design.size());
      gamma.resize(K);
      gamma = 0.0;
      if (K == 0) return max_element(nda::abs(target));

      nda::matrix<double> D(n * n, K);
      nda::vector<double> y(n * n);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
          y(i * n + j) = target(i, j);
          for (int k = 0; k < K; ++k) D(i * n + j, k) = design[k](i, j);
        }

      auto [U, s, Vt]  = nda::linalg::svd(D);
      double s_max     = (s.size() > 0) ? s(0) : 0.0;
      double threshold = 1.e-9 * std::max(s_max, 1.0);

      for (int k = 0; k < s.size(); ++k) {
        if (s(k) < threshold) continue;
        double Uty = 0.0;
        for (int r = 0; r < n * n; ++r) Uty += U(r, k) * y(r);
        double c = Uty / s(k);
        for (int col = 0; col < K; ++col) gamma(col) += Vt(k, col) * c;
      }

      double max_residual = 0.0;
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
          double fit = 0.0;
          for (int k = 0; k < K; ++k) fit += gamma(k) * design[k](i, j);
          max_residual = std::max(max_residual, std::abs(target(i, j) - fit));
        }
      return max_residual;
    }

  } // namespace

  void recover_conserved_density_groups(classified_dyn_vertices_t &classified, std::vector<nda::vector<double>> const &conserved_combinations,
                                        fundamental_operator_set const &fops, std::map<std::pair<int, int>, int> const &linindex) {
    if (conserved_combinations.empty() || classified.stochastic.empty()) return;

    // Density-bilinear candidates among the rejected (stochastic) vertices -- both
    // off-diagonal (a!=b) and diagonal (a==b) self-terms are eligible to participate.
    std::vector<size_t> candidate_indices;
    for (size_t i = 0; i < classified.stochastic.size(); ++i) {
      auto bp1 = extract_bilinear(classified.stochastic[i].op1, fops, linindex, "op1");
      auto bp2 = extract_bilinear(classified.stochastic[i].op2, fops, linindex, "op2");
      if (is_density_bilinear(bp1) && is_density_bilinear(bp2)) candidate_indices.push_back(i);
    }
    if (candidate_indices.empty()) return;

    // Group by shared coupling curve.
    std::vector<std::vector<size_t>> groups;
    for (auto i : candidate_indices) {
      bool placed = false;
      for (auto &g : groups)
        if (gf_close(classified.stochastic[i].coupling, classified.stochastic[g.front()].coupling)) {
          g.push_back(i);
          placed = true;
          break;
        }
      if (!placed) groups.push_back({i});
    }

    std::set<size_t> to_recover;
    for (auto const &g : groups) {
      std::map<std::pair<int, int>, size_t> pair_to_vertex;
      std::set<int> orbitals;
      for (auto i : g) {
        auto bp1 = extract_bilinear(classified.stochastic[i].op1, fops, linindex, "op1");
        auto bp2 = extract_bilinear(classified.stochastic[i].op2, fops, linindex, "op2");
        pair_to_vertex[{bp1.opL.linear_index, bp2.opL.linear_index}] = i;
        orbitals.insert(bp1.opL.linear_index);
        orbitals.insert(bp2.opL.linear_index);
      }
      if (orbitals.size() < 2) continue;

      // Require EVERY (a,b) pair among the touched orbitals, INCLUDING a==b, to be
      // present as its own vertex -- nothing inferred, see the module notes above.
      std::vector<int> S(orbitals.begin(), orbitals.end());
      bool complete = true;
      for (int a : S) {
        for (int b : S)
          if (!pair_to_vertex.count({a, b})) { complete = false; }
        if (!complete) break;
      }
      if (!complete) continue;

      int n = static_cast<int>(S.size());
      nda::matrix<double> target(n, n);
      // Uniform, in units of the group's shared coupling curve. NOTE: nda::matrix's
      // scalar assignment means "scalar * identity", not element-wise fill -- an
      // explicit loop is required for a genuinely all-ones matrix.
      for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q) target(p, q) = 1.0;

      std::vector<nda::matrix<double>> design;
      auto restrict_to_S = [&](nda::vector<double> const &O) {
        nda::vector<double> O_S(n);
        for (int k = 0; k < n; ++k) O_S(k) = O(S[k]);
        return O_S;
      };
      std::vector<nda::vector<double>> O_S_list;
      for (auto const &O : conserved_combinations) O_S_list.push_back(restrict_to_S(O));
      for (size_t bi = 0; bi < O_S_list.size(); ++bi) {
        nda::matrix<double> outer_ii(n, n);
        for (int p = 0; p < n; ++p)
          for (int q = 0; q < n; ++q) outer_ii(p, q) = O_S_list[bi](p) * O_S_list[bi](q);
        design.push_back(outer_ii);
        for (size_t bj = bi + 1; bj < O_S_list.size(); ++bj) {
          nda::matrix<double> outer_ij(n, n);
          for (int p = 0; p < n; ++p)
            for (int q = 0; q < n; ++q) outer_ij(p, q) = O_S_list[bi](p) * O_S_list[bj](q) + O_S_list[bj](p) * O_S_list[bi](q);
          design.push_back(outer_ij);
        }
      }

      nda::vector<double> gamma;
      double residual = fit_and_residual(design, target, n, gamma);
      if (residual > 1.e-8) continue; // not exactly representable -- leave as stochastic

      for (int a : S)
        for (int b : S) to_recover.insert(pair_to_vertex.at({a, b}));
    }

    if (to_recover.empty()) return;
    std::vector<dyn_vertex_t> remaining_stochastic;
    for (size_t i = 0; i < classified.stochastic.size(); ++i) {
      if (to_recover.count(i))
        classified.lang_firsov.push_back(classified.stochastic[i]);
      else
        remaining_stochastic.push_back(classified.stochastic[i]);
    }
    classified.stochastic = std::move(remaining_stochastic);
  }

} // namespace triqs_cthyb
