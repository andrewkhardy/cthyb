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

#include "./swap_dyn.hpp"

namespace triqs_cthyb {

  namespace {
    // One bilinear of a vertex with its time: leg 0 is (op1, tau1), leg 1 is (op2, tau2)
    struct leg_t {
      op_desc_pair_t op;
      time_pt tau;
    };
    leg_t leg(configuration::dyn_bosonic_pair_t const &v, int i) { return i == 0 ? leg_t{v.ops.op1, v.tau1} : leg_t{v.ops.op2, v.tau2}; }
  } // namespace

  move_swap_dyn::move_swap_dyn(qmc_data &data, mc_tools::random_generator &rng) : data(data), config(data.config), rng(rng) {}

  mc_weight_t move_swap_dyn::attempt() {

    auto const &vertices = config.dyn_oplist;
    if (vertices.size() < 2) return 0;

    // Two different vertices, and one leg of each
    index1 = rng(vertices.size());
    index2 = rng(vertices.size() - 1);
    if (index2 >= index1) ++index2;
    int const l1 = rng(2), l2 = rng(2);
    auto const &v1 = vertices[index1];
    auto const &v2 = vertices[index2];

    // Legs carrying the same bilinear trade partners, so no operator moves. Ambiguous types are never swapped.
    if (leg(v1, l1).op != leg(v2, l2).op) return 0;
    if (data.find_dyn_type(v1.ops.op1, v1.ops.op2) == -1 || data.find_dyn_type(v2.ops.op1, v2.ops.op2) == -1) return 0;

    // The vertex with legs a and b, op1 at the later time, if the catalog has its type
    auto make_vertex = [&](leg_t a, leg_t b, configuration::dyn_bosonic_pair_t &v) {
      if (a.tau < b.tau) std::swap(a, b);
      int const type = data.find_dyn_type(a.op, b.op);
      if (type == -1) return false;
      v = {data.dyn_op_list[type], a.tau, b.tau};
      return true;
    };
    if (!make_vertex(leg(v2, l2), leg(v1, 1 - l1), new_vertex1) || !make_vertex(leg(v1, l1), leg(v2, 1 - l2), new_vertex2)) return 0;

    // Trace, determinants and Lang-Firsov weight are unchanged: only the couplings differ
    double const old_couplings = data.dyn_coupling(v1) * data.dyn_coupling(v2);
    if (old_couplings == 0.0) return 0;
    return data.dyn_coupling(new_vertex1) * data.dyn_coupling(new_vertex2) / old_couplings;
  }

  mc_weight_t move_swap_dyn::accept() {
    config.dyn_oplist[index1] = new_vertex1;
    config.dyn_oplist[index2] = new_vertex2;
    config.finalize();
    return 1.0;
  }

  void move_swap_dyn::reject() { config.finalize(); }

} // namespace triqs_cthyb
