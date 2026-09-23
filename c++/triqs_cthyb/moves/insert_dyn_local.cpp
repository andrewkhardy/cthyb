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

// Local insertion of a stochastic dynamical vertex.
//
// move_insert_dyn draws tau1 and tau2 independently on [0, beta). For a spin-flip vertex
// S+(tau1) S-(tau2) the trace is non-zero only if nothing between the two times undoes the
// flip, i.e. (for a single orbital) only if both land in one operator-free stretch where the
// impurity is singly occupied. At beta = 100 with ~90 operators on the circle almost every
// independent draw fails that, and the move accepted 0.23% of proposals.
//
// This move draws tau_a uniformly on [0, beta), then tau_b uniformly in the operator-free arc
// of the trace containing tau_a, of length l. The weight is computed exactly as in
// move_insert_dyn; only the proposal differs. It is added *alongside* the global pair, so
// ergodicity is untouched.
//
// Detailed balance. Call a vertex *local* if, with its own four operators left out, both of
// its bilinears lie in one operator-free arc of the trace (qmc_data::local_vertex_gap). For a
// local vertex, taking either end first gives the same arc, so the proposal density of the
// unordered pair is
//     P_ins(x -> y) = (1 / N_types) * 2 / (beta l),
// with l that arc's length in x. The reverse move (move_remove_dyn_local) picks uniformly
// among the N_loc(y) local vertices of y, P_rem(y -> x) = 1 / N_loc(y), and finds the same arc
// (y with the vertex left out is x). Hence
//     P_rem / P_ins = beta l N_types / (2 N_loc(y)),
// which for l = beta and N_loc = n + 1 is exactly move_insert_dyn's t_ratio -- a check on the
// normalisation. N_loc(y) counts the new vertex and every existing vertex still local once the
// new one is in the trace (inserting can make a neighbour non-local).

#include "./insert_dyn_local.hpp"
#include <triqs/utility/time_pt.hpp>

namespace triqs_cthyb {

  move_insert_dyn_local::move_insert_dyn_local(qmc_data &data, mc_tools::random_generator &rng)
     : data(data), config(data.config), rng(rng) {}

  mc_weight_t move_insert_dyn_local::attempt() {

    // First end uniform on the circle, second uniform in the operator-free arc containing it
    auto tau_a = data.tau_seg.get_random_pt(rng);
    auto gap   = data.trace_gap(tau_a);
    auto tau_b = gap.lo + data.tau_seg.get_random_pt(rng, gap.length);
    if (tau_a == tau_b || !gap.contains(tau_b)) return 0; // measure-zero edges
    tau1 = std::max(tau_a, tau_b);
    tau2 = std::min(tau_a, tau_b);

    // Pick up pair of operators to insert
    dyn_pair = data.dyn_op_list[rng(data.dyn_op_list.size())];

    // Local vertices of the proposed configuration: the new one, and every existing one that
    // stays local with the new vertex's operators in the trace
    configuration::dyn_bosonic_pair_t const new_vertex{dyn_pair, tau1, tau2};
    long n_local_after = 1;
    for (long k = 0; k < long(config.dyn_oplist.size()); ++k)
      if (data.local_vertex_gap(k, &new_vertex) > 0.0) ++n_local_after;

    // Insert operators in the tree
    auto vertex_ops = data.dyn_vertex_ops(dyn_pair, tau1, tau2);
    try {
      for (auto const &[tau, op] : vertex_ops) data.imp_trace.try_insert(tau, op);
    } catch (rbt_insert_error const &) { // two operators at the same time point
      data.imp_trace.cancel_insert();
      return 0;
    }

    // Weight: the same three factors as move_insert_dyn
    double dyn_term_ratio    = -1 * data.dyn_interactions[dyn_pair.f_index](double(tau1 - tau2));
    double lang_firsov_ratio = data.compute_lang_firsov_ratio(vertex_ops, {});

    // Proposal probability ratio (see the header of this file)
    mc_weight_t t_ratio = config.beta() * double(gap.length) * double(data.dyn_op_list.size()) / (2.0 * double(n_local_after));

    // For quick abandon
    double random_number = rng.preview();
    if (random_number == 0.0) return 0;
    double p_yee = std::abs(t_ratio * dyn_term_ratio * lang_firsov_ratio / data.atomic_weight);

    // computation of the new trace after insertion
    std::tie(new_atomic_weight, new_atomic_reweighting) = data.imp_trace.compute(p_yee, random_number);
    if (new_atomic_weight == 0.0) { return 0; }
    auto atomic_weight_ratio = new_atomic_weight / data.atomic_weight;
    if (!isfinite(atomic_weight_ratio))
      TRIQS_RUNTIME_ERROR << "(insert_dyn_local) trace_ratio not finite " << new_atomic_weight << " " << data.atomic_weight << " "
                          << new_atomic_weight / data.atomic_weight << " in config " << config.get_id();

    mc_weight_t p = atomic_weight_ratio * dyn_term_ratio * lang_firsov_ratio;
    if (!isfinite(p * t_ratio))
      TRIQS_RUNTIME_ERROR << "(insert_dyn_local) p * t_ratio not finite p : " << p << " t_ratio : " << t_ratio << " in config "
                          << config.get_id();
    return p * t_ratio;
  }

  // -------------------------------------------------------------

  mc_weight_t move_insert_dyn_local::accept() {
    data.imp_trace.confirm_insert();
    config.dyn_oplist.push_back({dyn_pair, tau1, tau2});
    config.finalize();

    data.update_sign();
    data.atomic_weight      = new_atomic_weight;
    data.atomic_reweighting = new_atomic_reweighting;
    return data.current_sign / data.old_sign;
  }

  // ----------------------------------------

  void move_insert_dyn_local::reject() {
    config.finalize();
    data.imp_trace.cancel_insert();
  }
} // namespace triqs_cthyb
