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

#include <vector>
#include <set>
#include <algorithm>

namespace triqs_cthyb {

  class move_global {

    std::string name;

    qmc_data &data;
    configuration &config;
    mc_tools::random_generator &rng;

    // Substitutions as mappings (old linear index) -> (new op_desc)
    std::vector<op_desc> substitute_c, substitute_c_dag;

    // Indices of blocks potentially affected by this move
    std::set<int> affected_blocks;

    // Substitute every mapped operator, not a random subset of them
    bool full;

    // Hybridization operators to be updated, and all trace operators to be updated when dynamical vertices change too
    configuration::oplist_t updated_ops, updated_trace_ops;

    // Proposed dynamical vertices
    configuration::dyn_oplist_t new_dyn_oplist;

    // Proposed arguments of the dets
    std::vector<std::vector<det_type::x_type>> x;
    std::vector<std::vector<det_type::y_type>> y;

    h_scalar_t new_atomic_weight;      // Proposed value of the trace or norm
    h_scalar_t new_atomic_reweighting; // Proposed value of the reweighting

    public:
    move_global(std::string const &name, indices_map_t const &substitution_map, qmc_data &data, mc_tools::random_generator &rng, bool full);

    mc_weight_t attempt();
    mc_weight_t accept();
    void reject();
  };
}
