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
  // Total-density decomposition: classify_dyn_vertices above only ever accepts a
  // density vertex n_a-n_b for Lang-Firsov if n_a and n_b *individually* commute with
  // h_loc -- exactly what compute_lang_firsov_ratio's per-operator phase dressing
  // needs (see qmc_data.hpp). That's overly conservative for a genuine Hubbard-Kanamori
  // h_loc: spin-flip or pair-hopping terms break each n_a individually, even though the
  // *total* density N_total = sum_a n_a is still conserved (both terms move particles
  // between orbitals without changing the total count). When a group of vertices
  // couples uniformly and completely to every off-diagonal pair among a set of
  // orbitals, that whole group is exactly equivalent to a single coupling to
  // N_total^2, and can be resummed by Lang-Firsov even when the individual vertices
  // could not be -- see find_total_density_decomposition below for the precise
  // condition and apply_total_density_shift/apply_total_density_kernel for how it's
  // applied.
  // ---------------------------------------------------------------------------------

  struct total_density_decomposition_t {
    bool found = false;
    many_body_op_t total_density_op;           // N_total = sum of n_a over the orbitals in the group
    std::vector<int> orbital_linear_indices;   // the orbitals a that make up the group
    gf<imtime, scalar_valued> shared_coupling; // the single coupling D(tau) common to every off-diagonal pair
    std::vector<dyn_vertex_t> remaining_vertices; // every input vertex not absorbed into the group
  };

  /// Look for a "sufficiently symmetric" subset of `vertices`: two or more
  /// density-density vertices that (a) all share the exact same coupling D(tau), and
  /// (b) together cover *every* off-diagonal (a,b) pair among the orbitals they touch
  /// -- i.e. sum_{a != b in the group} D(tau) n_a(tau) n_b(0), complete and uniform.
  /// Under those two conditions this sum is exactly D(tau) * (N_total^2 - N_total)
  /// with N_total = sum_a n_a, regardless of whether the individual n_a commute with
  /// h_loc. Does not check commutation with h_loc itself (the caller does that, once,
  /// against total_density_op) -- this function is pure pattern-matching on the vertex
  /// list. If no such group exists, `found` is false and `remaining_vertices` is just
  /// `vertices` unchanged.
  total_density_decomposition_t find_total_density_decomposition(std::vector<dyn_vertex_t> const &vertices,
                                                                  fundamental_operator_set const &fops,
                                                                  std::map<std::pair<int, int>, int> const &linindex);

  /// Shift h_loc to account for the diagonal n_a^2 = n_a terms implicitly included when
  /// apply_total_density_kernel below couples every (a,b) pair uniformly (including
  /// a==b): without this correction the resummed interaction would be D(tau)*N_total^2
  /// instead of the intended D(tau)*sum_{a!=b} n_a n_b = D(tau)*(N_total^2 - N_total).
  /// Must run before h_diag is built from h_loc, alongside apply_lang_firsov_shift.
  void apply_total_density_shift(many_body_op_t &h_loc, total_density_decomposition_t const &decomposition, double beta, int N_leg,
                                 int verbosity);

  /// Populate K_n[a][b] with `decomposition.shared_coupling`'s Legendre coefficients for
  /// every pair (a,b) of orbitals in the group, including a==b -- qmc_data.hpp's
  /// compute_lang_firsov_ratio dresses every individual operator insertion at orbital a
  /// with a phase tied to K_n[a][b] against every other insertion at orbital b, so a
  /// uniform grid here reproduces a single coupling to N_total exactly (see the module
  /// notes above). Resizes K_n if it isn't large enough yet, so this can run whether or
  /// not build_K_n has already populated other orbitals' entries.
  void apply_total_density_kernel(total_density_decomposition_t const &decomposition, std::vector<std::vector<std::vector<double>>> &K_n,
                                  std::map<std::pair<int, int>, int> const &linindex, double beta, int N_leg);

} // namespace triqs_cthyb
