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

#include "./global.hpp"

namespace triqs_cthyb {

  move_global::move_global(std::string const &name, indices_map_t const &substitution_map, qmc_data &data, mc_tools::random_generator &rng,
                           bool full)
     : name(name),
       data(data),
       config(data.config),
       rng(rng),
       substitute_c(data.linindex.size()),
       substitute_c_dag(data.linindex.size()),
       full(full),
       x(data.dets.size()),
       y(data.dets.size()) {

    auto const &fops = data.h_diag.get_fops();

    // Inverse of data.linindex
    std::vector<std::pair<int, int>> lin_to_block_inner(data.linindex.size());
    for (auto const &l : data.linindex) lin_to_block_inner[l.second] = l.first;

    bool identity = true;
    for (int lin = 0; lin < lin_to_block_inner.size(); ++lin) {
      int new_lin, new_block, new_inner;

      // Does operator with linear index lin have a mapping in substitution_map?
      auto it = std::find_if(std::begin(substitution_map), std::end(substitution_map),
                             [&fops, lin](indices_map_t::value_type const &kv) { return fops[kv.first] == lin; });

      // If it does not, it is substituted by itself (subst_linear = lin)
      new_lin = (it != std::end(substitution_map)) ? fops[it->second] : lin;
      std::tie(new_block, new_inner) = lin_to_block_inner[new_lin];

      if (new_lin != lin) {
        identity = false;
        affected_blocks.insert(lin_to_block_inner[lin].first);
        affected_blocks.insert(new_block);
      }

      substitute_c[lin]     = op_desc{new_block, new_inner, false, new_lin};
      substitute_c_dag[lin] = op_desc{new_block, new_inner, true, new_lin};
    }

    if (identity) std::cerr << "WARNING: global move '" << name << "' changes no operator indices, therefore is useless." << std::endl;
  }

  mc_weight_t move_global::attempt() {

#ifdef EXT_DEBUG
    std::cerr << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
    std::cerr << "In config " << config.get_id() << std::endl;
    std::cerr << "* Attempt for move move_global (" << name << ")" << std::endl;
#endif

    updated_ops.clear();
    for (auto const &o : data.config) {
      auto const &tau    = o.first;
      auto const &old_op = o.second;
      auto const &new_op = (old_op.dagger ? substitute_c_dag : substitute_c)[old_op.linear_index];
      if (old_op.linear_index != new_op.linear_index) updated_ops.emplace(tau, new_op);
    }

#ifdef EXT_DEBUG
    std::cerr << updated_ops.size() << " out of " << data.config.size() << " operators can be changed" << std::endl;
#endif

    // Choose a random number of operators, which will not actually be updated
    // (we always update at least one operator) -- unless every mapped operator is substituted
    if (!full && updated_ops.size()) {
      int n_no_update = rng(updated_ops.size());
      // Remove some operators
      for (int i = 0; i < n_no_update; ++i) {
        auto it = std::begin(updated_ops);
        std::advance(it, rng(updated_ops.size()));
        updated_ops.erase(it);
      }
    }

#ifdef EXT_DEBUG
    std::cerr << updated_ops.size() << " operators will actually be changed" << std::endl;
#endif

    // Stochastic dynamical vertices. Their operators are in the trace but not in config, so
    // they are substituted here -- always as whole vertices, since a vertex with only some of
    // its four operators substituted is not a vertex of the catalog. The substituted operator
    // pair must itself be a catalog entry (for a spin flip, S+S- <-> S-S+), otherwise the
    // substitution is not a symmetry of the dynamical interaction and the move is rejected.
    // Times are unchanged, so op1 stays at the later time as the catalog form requires.
    updated_trace_ops = updated_ops;
    new_dyn_oplist    = config.dyn_oplist;
    dyn_changed       = false;
    double dyn_ratio  = 1.0;
    std::vector<std::pair<time_pt, op_desc>> dyn_inserted, dyn_removed;
    auto substitute   = [&](op_desc const &op) { return (op.dagger ? substitute_c_dag : substitute_c)[op.linear_index]; };
    for (auto &vertex : new_dyn_oplist) {
      op_desc_pair_t const op1{substitute(vertex.ops.op1.opL), substitute(vertex.ops.op1.opR)};
      op_desc_pair_t const op2{substitute(vertex.ops.op2.opL), substitute(vertex.ops.op2.opR)};
      if (op1 == vertex.ops.op1 && op2 == vertex.ops.op2) continue;

      int entry = -1;
      for (int i = 0; i < int(data.dyn_op_list.size()); ++i) {
        if (data.dyn_op_list[i].op1 == op1 && data.dyn_op_list[i].op2 == op2) {
          if (entry != -1) return 0; // registered twice: ambiguous, reject (as move_swap_dyn does)
          entry = i;
        }
      }
      if (entry == -1) return 0;

      double const dt        = double(vertex.tau1 - vertex.tau2);
      double const old_coupl = data.dyn_interactions[vertex.ops.f_index](dt);
      if (old_coupl == 0.0) return 0;
      dyn_ratio *= data.dyn_interactions[data.dyn_op_list[entry].f_index](dt) / old_coupl;

      auto const old_ops = data.dyn_vertex_ops(vertex.ops, vertex.tau1, vertex.tau2);
      vertex.ops         = data.dyn_op_list[entry];
      auto const new_ops = data.dyn_vertex_ops(vertex.ops, vertex.tau1, vertex.tau2);
      for (int m = 0; m < 4; ++m) {
        if (old_ops[m].second == new_ops[m].second) continue;
        updated_trace_ops.emplace(new_ops[m].first, new_ops[m].second);
        dyn_removed.push_back(old_ops[m]);
        dyn_inserted.push_back(new_ops[m]);
      }
      dyn_changed = true;
    }

    // No operators can be updated...
    if (updated_ops.empty() && !dyn_changed) return 0;

    // Derive new arguments of the dets
    for (auto block_index : affected_blocks) {
      x[block_index].clear();
      y[block_index].clear();
    }

    for (auto const &o : data.config) {
      auto const &tau    = o.first;
      auto it            = updated_ops.find(tau);
      auto const &new_op = it == updated_ops.end() ? o.second : it->second;
      (new_op.dagger ? x : y)[new_op.block_index].emplace_back(tau, new_op.inner_index);
    }

    for (auto block_index : affected_blocks) {
      // New configuration is not compatible with gf_struct
      if (x[block_index].size() != y[block_index].size()) return 0;
    }

    // Try refill determinants
    mc_weight_t det_ratio = 1;
    for (auto block_index : affected_blocks) {
      auto &det                   = data.dets[block_index];
      mc_weight_t block_det_ratio = det.try_refill(x[block_index], y[block_index]);
      if (block_det_ratio == .0) {
#ifdef EXT_DEBUG
        std::cerr << "block_det_ratio[" << block_index << "] = 0" << std::endl;
#endif
        return 0;
      }
      det_ratio *= block_det_ratio;
    }

      std::vector<std::pair<time_pt, op_desc>> inserted = dyn_inserted, removed = dyn_removed;
      for (auto const &o : updated_ops) {
        auto it = data.config.find(o.first);
        removed.push_back({it->first, it->second});
        inserted.push_back({o.first, o.second});
      }
      double lang_firsov_ratio = data.compute_lang_firsov_ratio(inserted, removed);

      // For quick abandon
      double random_number = rng.preview();
      if (random_number == 0.0) return 0;
      double p_yee = std::abs(det_ratio * lang_firsov_ratio * dyn_ratio / data.atomic_weight);

      data.imp_trace.try_replace(updated_trace_ops);

      // computation of the new trace after insertion
      std::tie(new_atomic_weight, new_atomic_reweighting) = data.imp_trace.compute(p_yee, random_number);
      if (new_atomic_weight == 0.0) {
  #ifdef EXT_DEBUG
        std::cerr << "trace == 0" << std::endl;
  #endif
        return 0;
      }
      auto atomic_weight_ratio = new_atomic_weight / data.atomic_weight;
      if (!isfinite(atomic_weight_ratio))
        TRIQS_RUNTIME_ERROR << "atomic_weight_ratio not finite " << new_atomic_weight << " " << data.atomic_weight << " "
                            << new_atomic_weight / data.atomic_weight << " in config " << config.get_id();

      mc_weight_t p = atomic_weight_ratio * det_ratio * lang_firsov_ratio * dyn_ratio;
#ifdef EXT_DEBUG    std::cerr << "Trace ratio: " << atomic_weight_ratio << '\t';
    std::cerr << "Det ratio: " << det_ratio << '\t';
    std::cerr << "p_yee: " << p_yee << std::endl;
    std::cerr << "Weight: " << p << std::endl;
#endif

    return p;
  }

  mc_weight_t move_global::accept() {

    for (auto const &o : updated_ops) data.config.replace(o.first, o.second);
    if (dyn_changed) config.dyn_oplist = new_dyn_oplist;
    config.finalize();

    for (auto block_index : affected_blocks) data.dets[block_index].complete_operation();

    data.update_sign();
    data.atomic_weight      = new_atomic_weight;
    data.atomic_reweighting = new_atomic_reweighting;

    data.imp_trace.confirm_replace();

#ifdef EXT_DEBUG
    std::cerr << "* Move move_global '" << name << "' accepted" << std::endl;
    std::cerr << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    for (int block_index : affected_blocks) check_det_sequence(data.dets[block_index], config.get_id());
#endif

    return data.current_sign / data.old_sign;
  }

  void move_global::reject() {

    config.finalize();
    data.imp_trace.cancel_replace();
    for (auto block_index : affected_blocks) data.dets[block_index].reject_last_try();

#ifdef EXT_DEBUG
    std::cerr << "* Move move_global '" << name << "' rejected" << std::endl;
    std::cerr << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    for (int block_index : affected_blocks) check_det_sequence(data.dets[block_index], config.get_id());
#endif
  }
}
