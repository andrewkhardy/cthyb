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

  total_density_decomposition_t find_total_density_decomposition(std::vector<dyn_vertex_t> const &vertices,
                                                                  fundamental_operator_set const &fops,
                                                                  std::map<std::pair<int, int>, int> const &linindex) {
    total_density_decomposition_t result;
    result.remaining_vertices = vertices;

    // Only genuine density-density vertices n_a-n_b (a != b) can be part of a
    // total-density group; a==b would be a diagonal self-term, not an off-diagonal pair.
    std::vector<size_t> candidate_indices;
    for (size_t i = 0; i < vertices.size(); ++i) {
      auto bp1 = extract_bilinear(vertices[i].op1, fops, linindex, "op1");
      auto bp2 = extract_bilinear(vertices[i].op2, fops, linindex, "op2");
      if (is_density_bilinear(bp1) && is_density_bilinear(bp2) && bp1.opL.linear_index != bp2.opL.linear_index) candidate_indices.push_back(i);
    }
    if (candidate_indices.size() < 2) return result; // need a genuine group, not a single pair

    // "Sufficiently symmetric" means every candidate shares the exact same coupling --
    // one boson coupled uniformly to every orbital's density, not a mix (e.g. Kanamori's
    // U and U' being different values would fail this, and correctly falls back to the
    // per-vertex path in classify_dyn_vertices instead).
    gf<imtime, scalar_valued> const &shared_coupling = vertices[candidate_indices.front()].coupling;
    for (auto i : candidate_indices)
      if (!gf_close(vertices[i].coupling, shared_coupling)) return result;

    // The orbitals touched, and the complete set of off-diagonal pairs among them --
    // if any pair (a,b) with a,b both in the group is missing, the group isn't complete
    // and substituting N_total^2 would add coupling for a pair the user never asked for.
    std::set<int> orbitals;
    std::set<std::pair<int, int>> pairs_present;
    for (auto i : candidate_indices) {
      auto bp1 = extract_bilinear(vertices[i].op1, fops, linindex, "op1");
      auto bp2 = extract_bilinear(vertices[i].op2, fops, linindex, "op2");
      orbitals.insert(bp1.opL.linear_index);
      orbitals.insert(bp2.opL.linear_index);
      pairs_present.insert({bp1.opL.linear_index, bp2.opL.linear_index});
    }
    size_t expected_pair_count = orbitals.size() * (orbitals.size() - 1);
    if (pairs_present.size() != expected_pair_count) return result;

    // Build N_total = sum of n_a over the group (reusing each vertex's own op1, one
    // representative per orbital, rather than reconstructing c_dag/c from indices), and
    // remove the absorbed vertices from the residual list.
    many_body_op_t total_density_op;
    std::set<int> seen_orbitals;
    for (auto i : candidate_indices) {
      auto bp1 = extract_bilinear(vertices[i].op1, fops, linindex, "op1");
      if (seen_orbitals.insert(bp1.opL.linear_index).second) total_density_op = total_density_op + vertices[i].op1;
    }

    result.found                  = true;
    result.total_density_op       = total_density_op;
    result.orbital_linear_indices = std::vector<int>(orbitals.begin(), orbitals.end());
    result.shared_coupling        = shared_coupling;
    result.remaining_vertices.clear();
    std::set<size_t> candidate_set(candidate_indices.begin(), candidate_indices.end());
    for (size_t i = 0; i < vertices.size(); ++i)
      if (!candidate_set.count(i)) result.remaining_vertices.push_back(vertices[i]);
    return result;
  }

  // -----------------------------------------------------------------------------------

  void apply_total_density_shift(many_body_op_t &h_loc, total_density_decomposition_t const &decomposition, double beta, int N_leg,
                                 int verbosity) {
    int n_pt_tau = decomposition.shared_coupling.mesh().size();
    auto d_n =
       fit_legendre_coeffs(n_pt_tau, beta, [&decomposition](double tau) { return eval_scalar_gf(decomposition.shared_coupling, tau); }, N_leg);
    double d0       = d_n(0);
    double d1       = (N_leg > 1) ? d_n(1) : 0.0;
    double Kprime_0 = -1.0 * beta * (d0 - d1 / 3.0);

    // apply_total_density_kernel below couples every (a,b) pair uniformly, including
    // a==b, which resums to D(tau)*N_total^2 = D(tau)*(N_total + sum_{a!=b} n_a n_b)
    // (using n_a^2 = n_a). Only the sum_{a!=b} part is wanted, so shift h_loc by +K'(0)
    // (opposite sign from apply_lang_firsov_shift's usual -0.5*K'(0)) to cancel the
    // extra D(tau)*N_total term.
    h_loc = h_loc + 0.5 * Kprime_0 * decomposition.total_density_op;

    if (verbosity >= 2)
      std::cout << "Lang-Firsov total-density decomposition: K'(0)=" << Kprime_0 << " diagonal correction for N_total = "
                << decomposition.total_density_op << std::endl;
  }

  // -----------------------------------------------------------------------------------

  void apply_total_density_kernel(total_density_decomposition_t const &decomposition, std::vector<std::vector<std::vector<double>>> &K_n,
                                  std::map<std::pair<int, int>, int> const &linindex, double beta, int N_leg) {
    auto M_matrix = build_M_matrix(N_leg, beta);
    int n_pt_tau  = decomposition.shared_coupling.mesh().size();
    auto d_n =
       fit_legendre_coeffs(n_pt_tau, beta, [&decomposition](double tau) { return eval_scalar_gf(decomposition.shared_coupling, tau); }, N_leg);
    nda::vector<double> k_n_vec = M_matrix * d_n;

    int max_linindex = 0;
    for (auto const &pair : linindex) max_linindex = std::max(max_linindex, pair.second);
    if (static_cast<int>(K_n.size()) < max_linindex + 1)
      K_n.resize(max_linindex + 1, std::vector<std::vector<double>>(max_linindex + 1, std::vector<double>(N_leg, 0.0)));

    for (int a : decomposition.orbital_linear_indices)
      for (int b : decomposition.orbital_linear_indices)
        for (int n = 0; n < N_leg; ++n) K_n[a][b][n] = k_n_vec(n);
  }

} // namespace triqs_cthyb
