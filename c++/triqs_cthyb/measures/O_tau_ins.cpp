/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2018, The Simons Foundation
 * Author: H. U.R. Strand
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

#include "./O_tau_ins.hpp"

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  measure_O_tau_ins::measure_O_tau_ins(std::optional<gf<imtime, scalar_valued>> &O_tau_opt, std::optional<gf<dlr, scalar_valued>> &O_dlr_opt,
                                       qmc_data const &data, dlr_imtime const &nodes, int n_tau, many_body_op_t const &op1,
                                       many_body_op_t const &op2)
     : sweep(data, nodes, {op2}, {op1}), nodes(nodes), symmetric((op1 - op2).is_zero()) {
    O_tau_opt = gf<imtime, scalar_valued>{{data.config.beta(), Boson, n_tau}};
    O_dlr_opt = gf<dlr, scalar_valued>{dlr{nodes}};
    O_tau.rebind(*O_tau_opt);
    O_dlr.rebind(*O_dlr_opt);
    O_tau() = 0.0;
    O_dlr() = 0.0;
  }

  void measure_O_tau_ins::collect_results(mpi::communicator const &c) {
    auto const corr = sweep.collect(c);
    auto at_nodes   = gf<dlr_imtime, scalar_valued>{nodes};
    for (auto const &tau : at_nodes.mesh()) {
      long const l  = tau.index();
      at_nodes[tau] = symmetric ? 0.5 * (corr(l, 0, 0) + corr(sweep.reflected(l), 0, 0)) : corr(l, 0, 0);
    }
    O_dlr = make_gf_dlr(at_nodes);
    for (auto const &tau : O_tau.mesh()) O_tau[tau] = O_dlr(tau.value());
  }

} // namespace triqs_cthyb
