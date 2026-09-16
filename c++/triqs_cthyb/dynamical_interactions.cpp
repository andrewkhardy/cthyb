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
#include <limits>
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
    h_scalar_t the_coeff = 0.0;
    for (auto const &[monomial, coeff] : op) {
      ++n_terms;
      the_monomial = monomial;
      the_coeff    = coeff;
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

    // A scalar prefactor on op1/op2 is redundant with the coupling -- D(tau) (a M1)(b M2) is the
    // same vertex as (a b D(tau)) M1 M2 -- and only the coupling is ever read here, so a factor
    // written on the operator would be dropped. Worse, it would be dropped *inconsistently*:
    // apply_lang_firsov_shift uses op1/op2 as full expressions, coefficients included, so the
    // analytic static shift and the sampled retarded part of the same vertex would disagree by
    // that factor. Refuse it rather than pick one interpretation silently.
    if (std::abs(the_coeff - 1.0) > 1.e-12)
      TRIQS_RUNTIME_ERROR << op_name << " carries the scalar coefficient " << the_coeff
                          << ", but a dynamical vertex is coupling(tau) * op1(tau) * op2(0): every numeric factor "
                             "belongs in the coupling, which is the only place it is read. Pass the bare bilinear "
                             "(n('up',0), not 0.5*n('up',0)) and multiply the coupling by the factor instead.";

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
                                                  bool lang_firsov_requested, bool debug) {
    auto commutes_with_hloc = [&](many_body_op_t const &op) { return (op * h_loc - h_loc * op).is_almost_zero(); };

    if (debug)
      std::cout << "\n[dyn_audit] classifying " << vertices.size() << " dynamical vertex(es)"
                << (lang_firsov_requested ? "" : " -- lang_firsov=false, so every one is forced stochastic") << ":\n";

    classified_dyn_vertices_t result;
    for (size_t i = 0; i < vertices.size(); ++i) {
      auto const &v = vertices[i];
      bool eligible = false;
      bool dens1 = false, dens2 = false, comm1 = false, comm2 = false;
      if (lang_firsov_requested) {
        auto bp1 = extract_bilinear(v.op1, fops, linindex, "op1");
        auto bp2 = extract_bilinear(v.op2, fops, linindex, "op2");
        dens1    = is_density_bilinear(bp1);
        dens2    = is_density_bilinear(bp2);
        comm1    = commutes_with_hloc(v.op1);
        comm2    = commutes_with_hloc(v.op2);
        eligible = dens1 && dens2 && comm1 && comm2;
      }
      if (debug) {
        auto yn = [](bool b) { return b ? "yes" : "no "; };
        std::cout << "[dyn_audit]   [" << i << "] " << v.op1 << "   (tau) x (0)   " << v.op2 << "\n"
                  << "[dyn_audit]        is n_a: " << yn(dens1) << " / " << yn(dens2)
                  << "   [op, h_loc] = 0: " << yn(comm1) << " / " << yn(comm2) << "   ->  "
                  << (eligible ? "Lang-Firsov (analytic)" : "stochastic") << "\n";
      }
      (eligible ? result.lang_firsov : result.stochastic).push_back(v);
    }
    return result;
  }

  // -----------------------------------------------------------------------------------

  lang_firsov_shift_t apply_lang_firsov_shift(many_body_op_t &h_loc, std::vector<dyn_vertex_t> const &lf_vertices,
                                              fundamental_operator_set const &fops, std::map<std::pair<int, int>, int> const &linindex,
                                              double beta, int N_leg, int verbosity) {
    if (lf_vertices.empty()) return {};

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

    return {U_renorm, mu_renorm};
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
      // Vertices on the same pair add up, as their static shifts do in apply_lang_firsov_shift
      for (int n = 0; n < N_leg; ++n) K_n[lin1][lin2][n] += k_n_vec(n);
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
    if (P == 0) { // every n_a commutes with h_loc individually: the whole space is conserved
      std::vector<nda::vector<double>> unit_vectors;
      for (int a = 0; a < M; ++a) {
        nda::vector<double> e(M);
        e    = 0.0;
        e(a) = 1.0;
        unit_vectors.push_back(e);
      }
      return unit_vectors;
    }

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

  conserved_densities_t conserved_densities(many_body_op_t const &h_loc, fundamental_operator_set const &fops,
                                            std::map<std::pair<int, int>, int> const &linindex) {
    auto basis = find_conserved_density_combinations(h_loc, fops, linindex);
    int r      = static_cast<int>(basis.size());
    if (r == 0) return {};
    int M = static_cast<int>(basis[0].size());

    // Reduced row-echelon form, with partial pivoting
    nda::matrix<double> A(r, M);
    for (int i = 0; i < r; ++i)
      for (int a = 0; a < M; ++a) A(i, a) = basis[i](a);
    int row = 0;
    for (int col = 0; col < M && row < r; ++col) {
      int pivot = row;
      for (int i = row + 1; i < r; ++i)
        if (std::abs(A(i, col)) > std::abs(A(pivot, col))) pivot = i;
      if (std::abs(A(pivot, col)) < 1.e-9) continue;
      for (int a = 0; a < M; ++a) std::swap(A(pivot, a), A(row, a));
      double const lead = A(row, col);
      for (int a = 0; a < M; ++a) A(row, a) /= lead;
      for (int i = 0; i < r; ++i) {
        if (i == row) continue;
        double const factor = A(i, col);
        for (int a = 0; a < M; ++a) A(i, a) -= factor * A(row, a);
      }
      ++row;
    }

    // n_a built from fops's own indices, as in find_conserved_density_combinations
    auto fops_indices = fundamental_operator_set::data_t(fops);
    conserved_densities_t result;
    for (int i = 0; i < r; ++i) {
      nda::vector<double> v(M);
      many_body_op_t op;
      for (int a = 0; a < M; ++a) {
        v(a) = (std::abs(A(i, a)) < 1.e-10) ? 0.0 : A(i, a);
        if (v(a) == 0.0) continue;
        op = op + v(a) * many_body_op_t::make_canonical(true, fops_indices[a]) * many_body_op_t::make_canonical(false, fops_indices[a]);
      }
      result.vectors.push_back(v);
      result.operators.push_back(op);
    }
    return result;
  }

  // -----------------------------------------------------------------------------------

  density_split_counts_t split_density_couplings(classified_dyn_vertices_t &classified, std::vector<nda::vector<double>> const &conserved_combinations,
                                                 fundamental_operator_set const &fops, std::map<std::pair<int, int>, int> const &linindex) {
    if (conserved_combinations.empty()) return {};

    // The density-density vertices classify_dyn_vertices rejected; every other vertex stays as it is
    std::vector<dyn_vertex_t> density_vertices, other_vertices;
    std::vector<std::pair<int, int>> density_pairs;
    for (auto const &v : classified.stochastic) {
      auto bp1 = extract_bilinear(v.op1, fops, linindex, "op1");
      auto bp2 = extract_bilinear(v.op2, fops, linindex, "op2");
      if (is_density_bilinear(bp1) && is_density_bilinear(bp2)) {
        density_vertices.push_back(v);
        density_pairs.emplace_back(bp1.opL.linear_index, bp2.opL.linear_index);
      } else
        other_vertices.push_back(v);
    }
    if (density_vertices.empty()) return {};

    // The Lang-Firsov part may only involve the conserved combinations. When these are indicator
    // vectors of disjoint orbital sets S_i (N, or N_up and N_down: the form conserved_densities returns
    // them in), that means a matrix constant on each block S_i x S_j and zero elsewhere. Otherwise the
    // rejected density vertices stay fully stochastic.
    int const M = count_orbitals(linindex);
    std::vector<std::vector<int>> members(conserved_combinations.size());
    std::vector<bool> assigned(M, false);
    for (size_t i = 0; i < conserved_combinations.size(); ++i) {
      for (int a = 0; a < M; ++a) {
        double const coefficient = conserved_combinations[i](a);
        if (std::abs(coefficient) < 1.e-10) continue;
        if (std::abs(coefficient - 1.0) > 1.e-10 || assigned[a]) return {static_cast<int>(density_vertices.size()), 0, 0, false};
        assigned[a] = true;
        members[i].push_back(a);
      }
    }

    // D_ab(tau) on the finest mesh among these vertices (others are read at the closest mesh point,
    // as the stochastic moves do); zero for pairs no vertex couples, vertices on one pair add up
    auto const *finest = &density_vertices[0].coupling;
    for (auto const &v : density_vertices)
      if (v.coupling.mesh().size() > finest->mesh().size()) finest = &v.coupling;
    auto const mesh = finest->mesh();
    int const n_tau = mesh.size();
    nda::array<double, 3> D(n_tau, M, M);
    D = 0.0;
    for (size_t k = 0; k < density_vertices.size(); ++k) {
      auto [a, b] = density_pairs[k];
      for (auto const &tau : mesh) D(tau.index(), a, b) += eval_scalar_gf(density_vertices[k].coupling, tau.value());
    }

    // Sign-preserving split, pointwise in tau (doc/notes/dynamical_interactions.tex, sec:split-sign):
    // on each block S_i x S_j the Lang-Firsov part is the entry of smallest magnitude when all entries
    // share a sign, and zero otherwise, so every residual entry keeps the sign of D or vanishes and no
    // stochastic vertex gets a weight of the opposite sign to the coupling it came from
    int const n_blocks = static_cast<int>(members.size());
    nda::array<double, 3> lang_firsov_part(n_tau, M, M), residual(n_tau, M, M);
    lang_firsov_part = 0.0;
    for (int t = 0; t < n_tau; ++t) {
      for (int i = 0; i < n_blocks; ++i) {
        for (int j = 0; j < n_blocks; ++j) {
          bool all_positive = true, all_negative = true;
          double smallest_magnitude = std::numeric_limits<double>::infinity(), closest_to_zero = 0.0;
          for (int a : members[i])
            for (int b : members[j]) {
              double const d = D(t, a, b);
              all_positive   = all_positive && d > 0.0;
              all_negative   = all_negative && d < 0.0;
              if (std::abs(d) < smallest_magnitude) {
                smallest_magnitude = std::abs(d);
                closest_to_zero    = d;
              }
            }
          double const lang_firsov_value = (all_positive || all_negative) ? closest_to_zero : 0.0;
          for (int a : members[i])
            for (int b : members[j]) lang_firsov_part(t, a, b) = lang_firsov_value;
        }
      }
    }
    residual = D - lang_firsov_part;

    // One vertex per pair with a non-zero entry; entries at round-off level (e.g. all of R when D is
    // exactly representable through the conserved combinations) are dropped
    double const threshold = 1.e-10 * max_element(nda::abs(D));
    auto fops_indices      = fundamental_operator_set::data_t(fops);
    auto density           = [&](int a) {
      return many_body_op_t::make_canonical(true, fops_indices[a]) * many_body_op_t::make_canonical(false, fops_indices[a]);
    };
    auto add_vertices = [&](nda::array<double, 3> const &coupling_matrix, std::vector<dyn_vertex_t> &out) {
      int added = 0;
      for (int a = 0; a < M; ++a) {
        for (int b = 0; b < M; ++b) {
          double largest = 0.0;
          for (int t = 0; t < n_tau; ++t) largest = std::max(largest, std::abs(coupling_matrix(t, a, b)));
          if (largest <= threshold) continue;
          auto coupling = gf<imtime, scalar_valued>{mesh};
          for (auto const &tau : mesh) coupling[tau] = coupling_matrix(tau.index(), a, b);
          out.push_back({density(a), density(b), coupling});
          ++added;
        }
      }
      return added;
    };

    density_split_counts_t counts;
    counts.n_input        = static_cast<int>(density_vertices.size());
    counts.n_lang_firsov  = add_vertices(lang_firsov_part, classified.lang_firsov);
    classified.stochastic = std::move(other_vertices);
    counts.n_stochastic   = add_vertices(residual, classified.stochastic);
    return counts;
  }

} // namespace triqs_cthyb
