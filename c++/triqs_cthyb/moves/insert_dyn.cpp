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

#include "./insert_dyn.hpp"

namespace triqs_cthyb {

  move_insert_dyn::move_insert_dyn(qmc_data &data, mc_tools::random_generator &rng) : data(data), config(data.config), rng(rng) {}

  mc_weight_t move_insert_dyn::attempt() {

    // Two times, op1 at the later one, then a vertex type
    vertex.tau1 = data.tau_seg.get_random_pt(rng);
    vertex.tau2 = data.tau_seg.get_random_pt(rng);
    if (vertex.tau1 < vertex.tau2) std::swap(vertex.tau1, vertex.tau2);
    vertex.ops = data.dyn_op_list[rng(data.dyn_op_list.size())];

    auto const ops = data.dyn_vertex_ops(vertex);
    try {
      for (auto const &[tau, op] : ops) data.imp_trace.try_insert(tau, op);
    } catch (rbt_insert_error const &) {
      std::cerr << "Insert error : recovering ... " << std::endl;
      data.imp_trace.cancel_insert();
      return 0;
    }

    double const beta        = config.beta();
    mc_weight_t t_ratio      = beta * beta / 2 * data.dyn_op_list.size() / double(config.dyn_oplist.size() + 1);
    double dyn_term_ratio    = -data.dyn_coupling(vertex);
    double lang_firsov_ratio = data.compute_lang_firsov_ratio(ops, {});

    // For quick abandon
    double random_number = rng.preview();
    if (random_number == 0.0) return 0;
    double p_yee = std::abs(t_ratio * dyn_term_ratio * lang_firsov_ratio / data.atomic_weight);

    std::tie(new_atomic_weight, new_atomic_reweighting) = data.imp_trace.compute(p_yee, random_number);
    if (new_atomic_weight == 0.0) return 0;

    mc_weight_t p = new_atomic_weight / data.atomic_weight * dyn_term_ratio * lang_firsov_ratio * t_ratio;
    if (!isfinite(p)) TRIQS_RUNTIME_ERROR << "(insert_dyn) weight ratio not finite: " << p << " in config " << config.get_id();
    return p;
  }

  mc_weight_t move_insert_dyn::accept() {
    data.imp_trace.confirm_insert();
    config.dyn_oplist.push_back(vertex);
    config.finalize();
    data.atomic_weight      = new_atomic_weight;
    data.atomic_reweighting = new_atomic_reweighting;
    return 1.0; // the permutation sign only involves hybridization operators
  }

  void move_insert_dyn::reject() {
    config.finalize();
    data.imp_trace.cancel_insert();
  }
} // namespace triqs_cthyb
