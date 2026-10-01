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
#pragma once
#include "./configuration.hpp"
#include "./types.hpp"
#include <triqs/hilbert_space/fundamental_operator_set.hpp>

// Dynamical (retarded) vertices D(tau) op1(tau) op2(0): collected from the solver inputs, routed either to the
// analytic Lang-Firsov resummation or to the stochastic expansion, and prepared for each.

namespace triqs_cthyb {

  /// The user's vertices, then one density vertex per non-zero entry of D0_tau, then the two spin flips of Jperp_tau
  /// (coupling Jperp/2 each; requires exactly two blocks of size 1)
  std::vector<dyn_vertex_t> collect_dyn_vertices(std::vector<dyn_vertex_t> const &explicit_vertices, block2_gf_const_view<imtime> D0t,
                                                 gf_const_view<imtime, matrix_valued> Jperpt, gf_struct_t const &gf_struct);

  struct classified_dyn_vertices_t {
    std::vector<dyn_vertex_t> lang_firsov, stochastic;
  };

  /// Lang-Firsov if requested and the vertex couples two densities that both commute with h_loc, stochastic otherwise
  classified_dyn_vertices_t classify_dyn_vertices(std::vector<dyn_vertex_t> const &vertices, many_body_op_t const &h_loc,
                                                  fundamental_operator_set const &fops, std::map<std::pair<int, int>, int> const &linindex,
                                                  bool lang_firsov_requested);

  /// The density combinations O_i = sum_a vectors[i][a] n_a that commute with h_loc, in reduced row-echelon form
  struct conserved_densities_t {
    std::vector<nda::vector<double>> vectors; // indexed by linear index
    std::vector<many_body_op_t> operators;
  };
  conserved_densities_t conserved_densities(many_body_op_t const &h_loc, fundamental_operator_set const &fops,
                                            std::map<std::pair<int, int>, int> const &linindex);

  /// Split the stochastic density vertices exactly, D = D_LF + R, into a Lang-Firsov part that couples only the conserved
  /// combinations (possible when they are indicators of disjoint orbital sets) and a residual R with the sign of D
  struct density_split_counts_t {
    int n_input = 0, n_lang_firsov = 0, n_stochastic = 0;
    bool block_structured = true; // false: the conserved combinations are not disjoint indicators, nothing was split
  };
  density_split_counts_t split_density_couplings(classified_dyn_vertices_t &classified, std::vector<nda::vector<double>> const &conserved_combinations,
                                                 fundamental_operator_set const &fops, std::map<std::pair<int, int>, int> const &linindex);

  /// Subtract the static part K'(0) of the Lang-Firsov vertices from h_loc, which must happen before h_diag is built
  void apply_lang_firsov_shift(many_body_op_t &h_loc, std::vector<dyn_vertex_t> const &lf_vertices, fundamental_operator_set const &fops,
                               std::map<std::pair<int, int>, int> const &linindex, double beta, int N_leg, int verbosity);

  /// Legendre coefficients K_n[a][b][n] of the Lang-Firsov kernel of qmc_data::compute_lang_firsov_ratio
  std::vector<std::vector<std::vector<double>>> build_K_n(std::vector<dyn_vertex_t> const &lf_vertices, double beta,
                                                          std::map<std::pair<int, int>, int> const &linindex, fundamental_operator_set const &fops,
                                                          int N_leg);

  /// The catalog of stochastic vertex types and their couplings, f_index = position
  void fold_into_stochastic_catalog(std::vector<dyn_vertex_t> const &stoch_vertices, fundamental_operator_set const &fops,
                                    std::map<std::pair<int, int>, int> const &linindex, std::vector<bosonic_op_pair_t> &dyn_op_list,
                                    std::vector<std::function<double(double)>> &dyn_interactions);

} // namespace triqs_cthyb
