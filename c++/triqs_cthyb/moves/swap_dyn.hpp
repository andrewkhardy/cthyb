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
#pragma once
#include <triqs/mc_tools.hpp>
#include "../qmc_data.hpp"

namespace triqs_cthyb {

  // Exchange of same-kind bilinears between two stochastic dynamical vertices (cf. ctseg's swap_spin_lines).
  // insert_dyn/remove_dyn alone cannot reach pairings in which every vertex crosses another.
  class move_swap_dyn {

    qmc_data &data;
    configuration &config;
    mc_tools::random_generator &rng;
    int index1, index2;
    configuration::dyn_bosonic_pair_t new_vertex1, new_vertex2;

    public:
    move_swap_dyn(qmc_data &data, mc_tools::random_generator &rng);

    mc_weight_t attempt();
    mc_weight_t accept();
    void reject();
  };
} // namespace triqs_cthyb
