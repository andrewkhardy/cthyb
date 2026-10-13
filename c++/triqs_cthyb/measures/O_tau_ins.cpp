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

  namespace {
    bool same(std::vector<many_body_op_t> const &x, std::vector<many_body_op_t> const &y) {
      if (x.size() != y.size()) return false;
      for (size_t i = 0; i < x.size(); ++i)
        if (!(x[i] - y[i]).is_zero()) return false;
      return true;
    }
  } // namespace

  measure_O_tau_ins::measure_O_tau_ins(std::optional<gf<imtime, matrix_valued>> &O_tau_opt, std::optional<gf<dlr, matrix_valued>> &O_dlr_opt,
                                       qmc_data const &data, dlr_imtime const &nodes, int n_tau, std::vector<many_body_op_t> const &ops1,
                                       std::vector<many_body_op_t> const &ops2)
     : sweep(data, nodes, ops2, ops1), nodes(nodes), symmetric(same(ops1, ops2)) {
    if (ops1.empty() or ops2.empty()) TRIQS_RUNTIME_ERROR << "measure_O_tau: empty list of operators";
    auto const shape = make_shape(long(ops2.size()), long(ops1.size()));
    O_tau_opt        = gf<imtime, matrix_valued>{imtime{data.config.beta(), Boson, n_tau}, shape};
    O_dlr_opt        = gf<dlr, matrix_valued>{dlr{nodes}, shape};
    O_tau.rebind(*O_tau_opt);
    O_dlr.rebind(*O_dlr_opt);
    O_tau() = 0.0;
    O_dlr() = 0.0;
  }

  void measure_O_tau_ins::collect_results(mpi::communicator const &c) {
    auto const corr = sweep.collect(c);
    auto at_nodes   = gf<dlr_imtime, matrix_valued>{nodes, O_tau.target_shape()};
    for (auto const &tau : at_nodes.mesh()) {
      long const l = tau.index();
      for (long i = 0; i < O_tau.target_shape()[0]; ++i)
        for (long j = 0; j < O_tau.target_shape()[1]; ++j)
          at_nodes[tau](i, j) = symmetric ? 0.5 * (corr(l, i, j) + corr(sweep.reflected(l), j, i)) : corr(l, i, j);
    }
    O_dlr = make_gf_dlr(at_nodes);
    for (auto const &tau : O_tau.mesh()) O_tau[tau] = O_dlr(tau.value());
  }

} // namespace triqs_cthyb
