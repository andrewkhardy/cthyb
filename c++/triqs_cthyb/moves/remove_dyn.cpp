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
#include <triqs/utility/time_pt.hpp>

namespace triqs_cthyb {

  move_remove_dyn::move_remove_dyn(qmc_data &data, mc_tools::random_generator &rng, histo_map_t *histos)
     : data(data), config(data.config), rng(rng) {}

  mc_weight_t move_remove_dyn::attempt() {

    // Check if there are any dynamical operators to remove
    if (config.dyn_oplist.size() == 0) return 0;

    // Pick up a random pair of dynamical operators to remove
    dyn_op_index      = rng(config.dyn_oplist.size());
    auto &dyn_op_pair = config.dyn_oplist[dyn_op_index];
    dyn_pair          = dyn_op_pair.ops;
    tau1              = dyn_op_pair.tau1;
    tau2              = dyn_op_pair.tau2;

    // Mark operators for deletion in the tree
    // Mirror the insertion pattern: we need to delete 4 operators (opL and opR for both op1 and op2)
    // These operators are in config.oplist, so we can count through config to find their indices

    data.imp_trace.try_delete(tau1);
    data.imp_trace.try_delete(tau2);
    data.imp_trace.try_delete(tau1 - data.tau_seg.get_epsilon());
    data.imp_trace.try_delete(tau2 - data.tau_seg.get_epsilon());

    //delete_op_pair(tau1, dyn_pair.op1);
    //delete_op_pair(tau2, dyn_pair.op2);

    // The ratio for the dynamic interaction (inverse of insertion)
    double dyn_term_ratio = 1.0 / data.dyn_interactions[dyn_pair.f_index](double(tau1 - tau2));

    // Proposal probability ratio (inverse of insertion)
    // Proposal probability ratio
    mc_weight_t reverse_probability = (2.0 / (config.beta() * config.beta())) * (1.0 / data.dyn_op_list.size());
    mc_weight_t direct_probability  = 1.0 / double(config.dyn_oplist.size());
    mc_weight_t t_ratio             = reverse_probability / direct_probability;

    // For quick abandon
    double random_number = rng.preview();
    if (random_number == 0.0) return 0;
    double p_yee = std::abs(t_ratio * dyn_term_ratio / data.atomic_weight);

    // computation of the new trace after removal
    std::tie(new_atomic_weight, new_atomic_reweighting) = data.imp_trace.compute(p_yee, random_number);
    if (new_atomic_weight == 0.0) { return 0; }
    auto atomic_weight_ratio = new_atomic_weight / data.atomic_weight;
    if (!isfinite(atomic_weight_ratio))
      TRIQS_RUNTIME_ERROR << "(remove_dyn) trace_ratio not finite " << new_atomic_weight << " " << data.atomic_weight << " "
                          << new_atomic_weight / data.atomic_weight << " in config " << config.get_id();

    mc_weight_t p = atomic_weight_ratio * dyn_term_ratio;

#ifdef EXT_DEBUG
    std::cerr << "Atomic ratio: " << atomic_weight_ratio << '\t';
    std::cerr << "Dyn term ratio: " << dyn_term_ratio << '\t';
    std::cerr << "Prefactor: " << t_ratio << '\t';
    std::cerr << "Weight: " << p * t_ratio << std::endl;
    std::cerr << "p_yee * newtrace: " << p_yee * new_atomic_weight << std::endl;
#endif

    if (!isfinite(p * t_ratio)) {
      std::cerr << "Remove_dyn move info:\n";
      std::cerr << "Atomic ratio: " << atomic_weight_ratio << '\t';
      std::cerr << "Dyn term ratio: " << dyn_term_ratio << '\t';
      std::cerr << "Prefactor: " << t_ratio << '\t';
      std::cerr << "Weight: " << p * t_ratio << std::endl;
      std::cerr << "p_yee * newtrace: " << p_yee * new_atomic_weight << std::endl;

      TRIQS_RUNTIME_ERROR << "(remove_dyn) p * t_ratio not finite p : " << p << " t_ratio : " << t_ratio << " in config " << config.get_id();
    }
    return p * t_ratio;
  }

  // -------------------------------------------------------------

  mc_weight_t move_remove_dyn::accept() {

    // remove from the tree
    data.imp_trace.confirm_delete();

    // remove from the configuration (all 4 operators)
    // config.erase(tau1);
    // config.erase(tau1 - data.tau_seg.get_epsilon());
    // config.erase(tau2);
    // config.erase(tau2 - data.tau_seg.get_epsilon());

    // Remove the pair of bosonic operators from the configuration
    config.dyn_oplist.erase(config.dyn_oplist.begin() + dyn_op_index);
    config.finalize();

    data.update_sign();
    data.atomic_weight      = new_atomic_weight;
    data.atomic_reweighting = new_atomic_reweighting;
  // if (data.current_sign/ data.old_sign != 1.0) {
  //     TRIQS_RUNTIME_ERROR << "(remove_dyn) Sign changed during bosonic operator removal! "
  //                         << "new sign is " << data.current_sign / data.old_sign
  //                         << " in config " << config.get_id();
  // }

    return data.current_sign / data.old_sign;
  }

  // ----------------------------------------

  void move_remove_dyn::reject() {

    config.finalize();
    data.imp_trace.cancel_delete();

#ifdef EXT_DEBUG
    std::cerr << "* Move move_remove_dyn rejected" << std::endl;
    std::cerr << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
#endif
  }
} // namespace triqs_cthyb