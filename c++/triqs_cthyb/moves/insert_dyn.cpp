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
#include <triqs/utility/time_pt.hpp>

namespace triqs_cthyb {

  move_insert_dyn::move_insert_dyn(qmc_data &data, mc_tools::random_generator &rng, histo_map_t *histos)
     : data(data), config(data.config), rng(rng) {}

  mc_weight_t move_insert_dyn::attempt() {

    // Choose 2 times tau1, tau2 for insertion
    tau1 = data.tau_seg.get_random_pt(rng);
    tau2 = data.tau_seg.get_random_pt(rng);
    if (tau1 < tau2) std::swap(tau1, tau2);

    // Pick up pair of operators to insert
    auto dyn_pair_idx = rng(data.dyn_op_list.size());
    dyn_pair          = data.dyn_op_list[dyn_pair_idx];

    // Insert operators in the tree
    try {
      auto insert_op_pair = [&](auto tau, const auto &op) {
        data.imp_trace.try_insert(tau, op.opL);
        data.imp_trace.try_insert(tau - data.tau_seg.get_epsilon(), op.opR);
      };
      insert_op_pair(tau1, dyn_pair.op1);
      insert_op_pair(tau2, dyn_pair.op2);
    } catch (rbt_insert_error const &) { // FIXME what this error ???
      std::cerr << "Insert error : recovering ... " << std::endl;
      data.imp_trace.cancel_insert();
      return 0;
    }

    // The ratio for the dynamic interaction
    double dyn_term_ratio = data.dyn_interactions[dyn_pair.f_index](double(tau1 - tau2));

    // Proposal probability ratio
    mc_weight_t direct_probability  = (2.0 / (config.beta() * config.beta())) * (1.0 / data.dyn_op_list.size());
    mc_weight_t reverse_probability = 1.0 / double(config.dyn_oplist.size() + 1);
    mc_weight_t t_ratio             = reverse_probability / direct_probability;

    // For quick abandon
    double random_number = rng.preview();
    if (random_number == 0.0) return 0;
    double p_yee = std::abs(t_ratio * dyn_term_ratio / data.atomic_weight);

    // computation of the new trace after insertion
    std::tie(new_atomic_weight, new_atomic_reweighting) = data.imp_trace.compute(p_yee, random_number);
    if (new_atomic_weight == 0.0) { return 0; }
    auto atomic_weight_ratio = new_atomic_weight / data.atomic_weight;
    if (!isfinite(atomic_weight_ratio))
      TRIQS_RUNTIME_ERROR << "(insert_dyn) trace_ratio not finite " << new_atomic_weight << " " << data.atomic_weight << " "
                          << new_atomic_weight / data.atomic_weight << " in config " << config.get_id();

    mc_weight_t p = atomic_weight_ratio * dyn_term_ratio;

#ifdef EXT_DEBUG
    std::cerr << "Atomic ratio: " << atomic_weight_ratio << '\t';
    std::cerr << "Det ratio: " << det_ratio << '\t';
    std::cerr << "Prefactor: " << t_ratio << '\t';
    std::cerr << "Weight: " << p * t_ratio << std::endl;
    std::cerr << "p_yee * newtrace: " << p_yee * new_atomic_weight << std::endl;
#endif

    if (!isfinite(p * t_ratio)) {
      std::cerr << "Insert_dyn move info:\n";
      std::cerr << "Atomic ratio: " << atomic_weight_ratio << '\t';
      std::cerr << "Det ratio: " << dyn_term_ratio << '\t';
      std::cerr << "Prefactor: " << t_ratio << '\t';
      std::cerr << "Weight: " << p * t_ratio << std::endl;
      std::cerr << "p_yee * newtrace: " << p_yee * new_atomic_weight << std::endl;

      TRIQS_RUNTIME_ERROR << "(insert_dyn) p * t_ratio not finite p : " << p << " t_ratio : " << t_ratio << " in config " << config.get_id();
    }
    return p * t_ratio;
  }

  // -------------------------------------------------------------

  mc_weight_t move_insert_dyn::accept() {

    // insert in the tree
    data.imp_trace.confirm_insert();

    //insert in the configuration (all 4 operators: opL and opR for both op1 and op2)
    config.insert(tau1, dyn_pair.op1.opL);
    config.insert(tau1 - data.tau_seg.get_epsilon(), dyn_pair.op1.opR);
    config.insert(tau2, dyn_pair.op2.opL);
    config.insert(tau2 - data.tau_seg.get_epsilon(), dyn_pair.op2.opR);
    
    // Insert the pair of bosonic operators in the configuration
    config.dyn_oplist.push_back({dyn_pair, tau1, tau2});
    config.finalize();

    data.update_sign();
    data.atomic_weight      = new_atomic_weight;
    data.atomic_reweighting = new_atomic_reweighting;
    // if (histo_accepted) *histo_accepted << dtau;
  // if (data.current_sign/ data.old_sign != 1.0) {
  //     TRIQS_RUNTIME_ERROR << "(insert_dyn) Sign changed during bosonic operator insertion! "
  //                         << "new sign is " << data.current_sign / data.old_sign
  //                         << " in config " << config.get_id();
  // }


    return data.current_sign / data.old_sign;
  }

  // ----------------------------------------

  void move_insert_dyn::reject() {

    config.finalize();
    data.imp_trace.cancel_insert();
    // data.dets[block_index].reject_last_try();

#ifdef EXT_DEBUG
    std::cerr << "* Move move_insert_dyn rejected" << std::endl;
    std::cerr << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    // check_det_sequence(data.dets[block_index], config.get_id());
#endif
  }
} // namespace triqs_cthyb