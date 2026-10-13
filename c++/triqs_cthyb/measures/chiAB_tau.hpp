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

  // chi_AB(tau)[i, j] = <A_i(tau) B_j(0)> for the lists chi_A_vec and chi_B_vec of bosonic operators, as CTINT's chiAB_tau:
  // measured at the DLR nodes, kept as DLR coefficients (chiAB_dlr) and evaluated on the regular mesh of n_tau points (chiAB_tau)
  class measure_chiAB_tau {

    public:
    measure_chiAB_tau(std::optional<gf<imtime, matrix_valued>> &chiAB_tau_opt, std::optional<gf<dlr, matrix_valued>> &chiAB_dlr_opt,
                      qmc_data const &data, dlr_imtime const &nodes, int n_tau, std::vector<many_body_op_t> const &A_vec,
                      std::vector<many_body_op_t> const &B_vec);
    void accumulate(mc_weight_t s) { sweep.accumulate(s); }
    void collect_results(mpi::communicator const &c);

    private:
    occupation_sweep sweep;
    dlr_imtime nodes;
    bool symmetric; // A = B, so that chi[i, j](tau) = chi[j, i](beta - tau)
    gf<imtime, matrix_valued>::view_type chiAB_tau;
    gf<dlr, matrix_valued>::view_type chiAB_dlr;
  };

} // namespace triqs_cthyb
