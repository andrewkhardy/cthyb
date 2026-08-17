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

// Everything a dynamical (retarded) interaction vertex needs, from the point a user
// registers it (explicitly, or via the D0_tau/Jperp_tau convenience inputs) through to
// the two mechanisms that can sample it: the analytic Lang-Firsov resummation (K_n /
// compute_lang_firsov_ratio in qmc_data.hpp) or the stochastic double expansion
// (dyn_op_list/dyn_interactions, sampled by moves/insert_dyn.cpp & remove_dyn.cpp).
//
// Nothing here guesses: which mechanism a vertex uses is decided by symbolic operator
// algebra against h_loc (see classify_dyn_vertices), not by the shape of the input data.

namespace triqs_cthyb {

  /// Validate that `op` reduces to exactly one fermion bilinear (one creation, one
  /// annihilation operator) in `fops`, and extract it as {opL, opR} with opL the
  /// creation operator. Throws a specific TRIQS_RUNTIME_ERROR (naming `op_name`,
  /// e.g. "op1") describing exactly what's wrong otherwise -- this is the validation
  /// that lets vertices be specified as ordinary many_body_operator expressions
  /// (e.g. c_dag('up',0)*c('down',0)) while still failing loudly on malformed input.
  op_desc_pair_t extract_bilinear(many_body_op_t const &op, fundamental_operator_set const &fops,
                                  std::map<std::pair<int, int>, int> const &linindex, std::string const &op_name);

  /// True if `bp` is a number operator n_a = c^dagger_a c_a for a single orbital a
  /// (opL and opR refer to the same block/inner index) -- the only shape the analytic
  /// Lang-Firsov resummation can represent (see the Double Expansion notes for why).
  bool is_density_bilinear(op_desc_pair_t const &bp);

  /// Expand D0_tau (block-pair x inner-index retarded density-density coupling) into
  /// vertices n_{bl1,i1}(tau) n_{bl2,i2}(0), appended to `vertices`. Every non-zero
  /// (bl1, i1, bl2, i2) entry becomes its own vertex; no orbital/spin convention is
  /// assumed since a density-density coupling doesn't need one.
  void expand_D0_into_vertices(block2_gf_const_view<imtime> D0t, gf_struct_t const &gf_struct, std::vector<dyn_vertex_t> &vertices);

  /// Expand Jperp_tau (spin-flip coupling) into vertices. Jperp_tau is a single
  /// global up/down coupling with no orbital index of its own (matches ctseg exactly),
  /// so this only supports the unambiguous case of exactly 2 blocks, each a single
  /// fermion mode. For per-orbital-pair or inter-orbital spin-flip (more blocks, or
  /// more orbitals per block), raises an error directing the user to
  /// solver_core::add_dyn_vertex instead of guessing a block-layout convention.
  void expand_Jperp_into_vertices(gf_const_view<imtime, matrix_valued> Jperpt, gf_struct_t const &gf_struct, std::vector<dyn_vertex_t> &vertices);

  /// Collect the full vertex list for a solve: the user's explicit vertices, plus
  /// D0_tau/Jperp_tau expanded into the same representation via the two functions above.
  std::vector<dyn_vertex_t> collect_dyn_vertices(std::vector<dyn_vertex_t> const &explicit_vertices, block2_gf_const_view<imtime> D0t,
                                                 gf_const_view<imtime, matrix_valued> Jperpt, gf_struct_t const &gf_struct);

  struct classified_dyn_vertices_t {
    std::vector<dyn_vertex_t> lang_firsov; // routed through the K_n / compute_lang_firsov_ratio path
    std::vector<dyn_vertex_t> stochastic;  // routed through insert_dyn/remove_dyn (dyn_op_list/dyn_interactions)
  };

  /// Split `vertices` into Lang-Firsov-eligible and stochastic-only. A vertex is
  /// Lang-Firsov-eligible only if `lang_firsov_requested` and it is a density-density
  /// coupling (both op1 and op2 are number operators -- a structural requirement of
  /// the polaron transform, not just a physical one) AND both op1 and op2 individually
  /// commute with h_loc (checked by plain operator algebra, op*h_loc - h_loc*op, not
  /// via atom_diag -- see the Double Expansion notes for why). Every vertex ends up in
  /// exactly one of the two lists, so nothing is ever silently dropped.
  classified_dyn_vertices_t classify_dyn_vertices(std::vector<dyn_vertex_t> const &vertices, many_body_op_t const &h_loc,
                                                  fundamental_operator_set const &fops, std::map<std::pair<int, int>, int> const &linindex,
                                                  bool lang_firsov_requested);

  /// Subtract the K'(0) static (instantaneous) part of each Lang-Firsov-eligible
  /// vertex's coupling from h_loc, so it isn't double-counted once the retarded part
  /// is resummed analytically. Must run before h_diag is built from h_loc. At
  /// verbosity>=2, also prints the aggregate before/after density-density interaction
  /// matrix (as in CTSEG), sized to all orbitals regardless of vertex count/source.
  void apply_lang_firsov_shift(many_body_op_t &h_loc, std::vector<dyn_vertex_t> const &lf_vertices, fundamental_operator_set const &fops,
                               std::map<std::pair<int, int>, int> const &linindex, double beta, int N_leg, int verbosity);

  /// Build the K_n[a][b][:] Legendre-coefficient kernel used by
  /// qmc_data::compute_lang_firsov_ratio, from the Lang-Firsov-eligible vertices.
  std::vector<std::vector<std::vector<double>>> build_K_n(std::vector<dyn_vertex_t> const &lf_vertices, double beta,
                                                          std::map<std::pair<int, int>, int> const &linindex,
                                                          fundamental_operator_set const &fops, int N_leg);

  /// Fold the stochastic vertices into the dyn_op_list/dyn_interactions catalog
  /// sampled by moves/insert_dyn.cpp and moves/remove_dyn.cpp.
  void fold_into_stochastic_catalog(std::vector<dyn_vertex_t> const &stoch_vertices, fundamental_operator_set const &fops,
                                    std::map<std::pair<int, int>, int> const &linindex, std::vector<bosonic_op_pair_t> &dyn_op_list,
                                    std::vector<std::function<double(double)>> &dyn_interactions);

  // ---------------------------------------------------------------------------------
  // Recovering vertices classify_dyn_vertices had to reject: a density vertex n_a-n_b
  // is only Lang-Firsov-eligible there if n_a AND n_b *individually* commute with
  // h_loc -- exactly what compute_lang_firsov_ratio's per-operator phase dressing
  // needs. That's correct but conservative: under a genuine Hubbard-Kanamori h_loc with
  // spin-flip/pair-hopping, no individual n_a commutes, even though *some combination*
  // of densities usually still does -- e.g. total charge N_total = sum_a n_a always
  // (spin-flip/pair-hopping only move particles between orbitals), and total S_z =
  // sum_a +-0.5 n_a whenever the Hamiltonian doesn't break spin-rotation symmetry.
  // find_conserved_density_combinations finds every such combination directly from
  // h_loc (not by guessing N_total specifically), and recover_conserved_density_groups
  // uses them to rescue whatever vertices classify_dyn_vertices correctly rejected but
  // that are, together, exactly equivalent to a coupling to one or more of those
  // combinations. See the .cpp for the operator-algebra argument for why this only
  // works when the group is *completely* user-specified (diagonal a==b self-terms
  // included, not silently inferred) -- a previous version of this mechanism only
  // required completeness among the off-diagonal pairs and inferred the rest, which was
  // an unsound approximation, not an exact resummation (it broke a genuine J*Sz*Sz
  // interaction, where the off-diagonal and diagonal couplings genuinely differ).
  // ---------------------------------------------------------------------------------

  /// Find a basis for the space of density-operator linear combinations that commute
  /// with h_loc: { c in R^M : [sum_a c_a n_a, h_loc] = 0 }, where a ranges over every
  /// orbital in linindex (M = total count). This is a genuine linear-algebra problem,
  /// not a pattern-match: the commutator is linear in c, so the solution set is exactly
  /// the nullspace of the linear map c -> sum_a c_a [n_a, h_loc]. Computed by expanding
  /// each [n_a, h_loc] via ordinary symbolic operator algebra (same primitive
  /// classify_dyn_vertices already uses), collecting every distinct monomial that
  /// appears across all M commutators as a basis for a real coefficient matrix (real
  /// and imaginary parts of each coefficient kept as separate rows, so this is correct
  /// whether h_scalar_t is real or complex), and taking that matrix's nullspace via
  /// SVD. Returns each basis vector as an nda::vector<double> of length M, indexed by
  /// linear index. Empty if no such combination exists (or if every n_a individually
  /// commutes already, a degenerate case classify_dyn_vertices already handles alone).
  std::vector<nda::vector<double>> find_conserved_density_combinations(many_body_op_t const &h_loc, fundamental_operator_set const &fops,
                                                                        std::map<std::pair<int, int>, int> const &linindex);

  /// Attempt to recover Lang-Firsov eligibility for vertices in `classified.stochastic`
  /// by checking whether a complete set of them exactly reconstructs a combination of
  /// `conserved_combinations`. Only ever recovers a group that is *completely*,
  /// explicitly specified: every ordered pair (a,b) among the touched orbitals,
  /// including the diagonal a==b self-terms, must already exist as its own vertex
  /// sharing one coupling curve -- nothing is inferred or filled in (a missing diagonal
  /// self-term is never silently added). If the fit against the conserved combinations
  /// is exact, the group's vertices (already fully specified, so nothing needs
  /// reconstructing) are moved from `classified.stochastic` to `classified.lang_firsov`
  /// in place; otherwise `classified` is left untouched for that group. The ordinary,
  /// unmodified apply_lang_firsov_shift/build_K_n handle everything recovered this way,
  /// exactly as if classify_dyn_vertices had accepted it directly.
  void recover_conserved_density_groups(classified_dyn_vertices_t &classified, std::vector<nda::vector<double>> const &conserved_combinations,
                                        fundamental_operator_set const &fops, std::map<std::pair<int, int>, int> const &linindex);

} // namespace triqs_cthyb
