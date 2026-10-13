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
#pragma once

#include <triqs/gfs.hpp>
#include <triqs/mesh.hpp>

#include "../qmc_data.hpp"
#include "./occupation_sweep.hpp"

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  // <O2[i](tau) O1[j](0)> for measure_O_tau = (O1, O2), lists of bosonic operators: measured at the DLR nodes, kept as DLR
  // coefficients (O_dlr) and evaluated on the regular mesh of n_tau points (O_tau)
  class measure_O_tau_ins {

    public:
    measure_O_tau_ins(std::optional<gf<imtime, matrix_valued>> &O_tau_opt, std::optional<gf<dlr, matrix_valued>> &O_dlr_opt, qmc_data const &data,
                      dlr_imtime const &nodes, int n_tau, std::vector<many_body_op_t> const &ops1, std::vector<many_body_op_t> const &ops2);
    void accumulate(mc_weight_t s) { sweep.accumulate(s); }
    void collect_results(mpi::communicator const &c);

    private:
    occupation_sweep sweep;
    dlr_imtime nodes;
    bool symmetric; // O1 = O2, so that O[i, j](tau) = O[j, i](beta - tau)
    gf<imtime, matrix_valued>::view_type O_tau;
    gf<dlr, matrix_valued>::view_type O_dlr;
  };

} // namespace triqs_cthyb
