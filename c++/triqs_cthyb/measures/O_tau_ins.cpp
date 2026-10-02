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

  measure_O_tau_ins::measure_O_tau_ins(std::optional<gf<imtime, scalar_valued>> &O_tau_opt, qmc_data const &data, int n_tau,
                                       many_body_op_t const &op1, many_body_op_t const &op2)
     : sweep(data, n_tau, {op2}, {op1}), symmetric((op1 - op2).is_zero()) {
    O_tau_opt = gf<imtime, scalar_valued>{{data.config.beta(), Boson, n_tau}};
    O_tau.rebind(*O_tau_opt);
    O_tau() = 0.0;
  }

  void measure_O_tau_ins::collect_results(mpi::communicator const &c) {
    auto const corr = sweep.collect(c);
    long const last = O_tau.mesh().size() - 1;
    for (auto const &tau : O_tau.mesh()) {
      long const p = tau.index();
      O_tau[tau]   = symmetric ? 0.5 * (corr(p, 0, 0) + corr(last - p, 0, 0)) : corr(p, 0, 0);
    }
  }

} // namespace triqs_cthyb
