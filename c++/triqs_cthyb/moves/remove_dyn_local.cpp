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

#include "./remove_dyn_local.hpp"
#include <triqs/utility/time_pt.hpp>

namespace triqs_cthyb {

  move_remove_dyn_local::move_remove_dyn_local(qmc_data &data, mc_tools::random_generator &rng)
     : data(data), config(data.config), rng(rng) {}

  mc_weight_t move_remove_dyn_local::attempt() {

    // Candidates: the local vertices, each with the operator-free arc its insertion drew from
    local_vertices.clear();
    local_gaps.clear();
    for (long k = 0; k < long(config.dyn_oplist.size()); ++k) {
      double gap = data.local_vertex_gap(k);
      if (gap > 0.0) {
        local_vertices.push_back(k);
        local_gaps.push_back(gap);
      }
    }
    if (local_vertices.empty()) return 0;

    auto pick           = rng(local_vertices.size());
    dyn_op_index        = local_vertices[pick];
    auto const &vertex  = config.dyn_oplist[dyn_op_index];
    auto const dyn_pair = vertex.ops;

    auto vertex_ops = data.dyn_vertex_ops(dyn_pair, vertex.tau1, vertex.tau2);
    for (auto const &vertex_op : vertex_ops) data.imp_trace.try_delete(vertex_op.first);

    // Weight: the same three factors as move_remove_dyn
    double dyn_term_ratio    = -1.0 / data.dyn_interactions[dyn_pair.f_index](double(vertex.tau1 - vertex.tau2));
    double lang_firsov_ratio = data.compute_lang_firsov_ratio({}, vertex_ops);

    // Proposal probability ratio, the inverse of move_insert_dyn_local's
    mc_weight_t t_ratio =
       2.0 * double(local_vertices.size()) / (config.beta() * local_gaps[pick] * double(data.dyn_op_list.size()));

    // For quick abandon
    double random_number = rng.preview();
    if (random_number == 0.0) return 0;
    double p_yee = std::abs(t_ratio * dyn_term_ratio * lang_firsov_ratio / data.atomic_weight);

    // computation of the new trace after removal
    std::tie(new_atomic_weight, new_atomic_reweighting) = data.imp_trace.compute(p_yee, random_number);
    if (new_atomic_weight == 0.0) { return 0; }
    auto atomic_weight_ratio = new_atomic_weight / data.atomic_weight;
    if (!isfinite(atomic_weight_ratio))
      TRIQS_RUNTIME_ERROR << "(remove_dyn_local) trace_ratio not finite " << new_atomic_weight << " " << data.atomic_weight << " "
                          << new_atomic_weight / data.atomic_weight << " in config " << config.get_id();

    mc_weight_t p = atomic_weight_ratio * dyn_term_ratio * lang_firsov_ratio;
    if (!isfinite(p * t_ratio))
      TRIQS_RUNTIME_ERROR << "(remove_dyn_local) p * t_ratio not finite p : " << p << " t_ratio : " << t_ratio << " in config "
                          << config.get_id();
    return p * t_ratio;
  }

  // -------------------------------------------------------------

  mc_weight_t move_remove_dyn_local::accept() {
    data.imp_trace.confirm_delete();
    config.dyn_oplist.erase(config.dyn_oplist.begin() + dyn_op_index);
    config.finalize();

    data.update_sign();
    data.atomic_weight      = new_atomic_weight;
    data.atomic_reweighting = new_atomic_reweighting;
    return data.current_sign / data.old_sign;
  }

  // ----------------------------------------

  void move_remove_dyn_local::reject() {
    config.finalize();
    data.imp_trace.cancel_delete();
  }
} // namespace triqs_cthyb
