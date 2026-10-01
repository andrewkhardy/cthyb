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

#include "./remove_dyn.hpp"

namespace triqs_cthyb {

  move_remove_dyn::move_remove_dyn(qmc_data &data, mc_tools::random_generator &rng) : data(data), config(data.config), rng(rng) {}

  mc_weight_t move_remove_dyn::attempt() {

    if (config.dyn_oplist.empty()) return 0;
    index              = rng(config.dyn_oplist.size());
    auto const &vertex = config.dyn_oplist[index];

    auto const ops = data.dyn_vertex_ops(vertex);
    for (auto const &op : ops) data.imp_trace.try_delete(op.first);

    double const beta        = config.beta();
    mc_weight_t t_ratio      = 2 / (beta * beta) * config.dyn_oplist.size() / double(data.dyn_op_list.size());
    double dyn_term_ratio    = -1.0 / data.dyn_coupling(vertex);
    double lang_firsov_ratio = data.compute_lang_firsov_ratio({}, ops);

    // For quick abandon
    double random_number = rng.preview();
    if (random_number == 0.0) return 0;
    double p_yee = std::abs(t_ratio * dyn_term_ratio * lang_firsov_ratio / data.atomic_weight);

    std::tie(new_atomic_weight, new_atomic_reweighting) = data.imp_trace.compute(p_yee, random_number);
    if (new_atomic_weight == 0.0) return 0;

    mc_weight_t p = new_atomic_weight / data.atomic_weight * dyn_term_ratio * lang_firsov_ratio * t_ratio;
    if (!isfinite(p)) TRIQS_RUNTIME_ERROR << "(remove_dyn) weight ratio not finite: " << p << " in config " << config.get_id();
    return p;
  }

  mc_weight_t move_remove_dyn::accept() {
    data.imp_trace.confirm_delete();
    config.dyn_oplist.erase(config.dyn_oplist.begin() + index); // keeps the order of the other vertices
    config.finalize();
    data.atomic_weight      = new_atomic_weight;
    data.atomic_reweighting = new_atomic_reweighting;
    return 1.0; // the permutation sign only involves hybridization operators
  }

  void move_remove_dyn::reject() {
    config.finalize();
    data.imp_trace.cancel_delete();
  }
} // namespace triqs_cthyb
