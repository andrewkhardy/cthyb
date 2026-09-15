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
#include <array>

namespace triqs_cthyb {

  namespace {
    // One bilinear of a vertex together with its time: slot 0 is (op1, tau1), slot 1 is (op2, tau2).
    struct timed_bilinear {
      op_desc_pair_t op;
      time_pt tau;
    };
  } // namespace

  move_swap_dyn::move_swap_dyn(qmc_data &data, mc_tools::random_generator &rng) : data(data), config(data.config), rng(rng) {}

  mc_weight_t move_swap_dyn::attempt() {

    auto &vertices = config.dyn_oplist;
    if (vertices.size() < 2) return 0;

    // Pick two different vertices, and one bilinear in each
    index1 = rng(vertices.size());
    index2 = rng(vertices.size() - 1);
    if (index2 >= index1) ++index2;
    auto const &vertex1 = vertices[index1];
    auto const &vertex2 = vertices[index2];

    auto bilinears = [](configuration::dyn_bosonic_pair_t const &v) {
      return std::array<timed_bilinear, 2>{timed_bilinear{v.ops.op1, v.tau1}, timed_bilinear{v.ops.op2, v.tau2}};
    };
    auto b1    = bilinears(vertex1);
    auto b2    = bilinears(vertex2);
    int slot1  = rng(2);
    int slot2  = rng(2);

    // Only bilinears of the same kind can trade partners: then every operator stays where it is
    if (!(b1[slot1].op == b2[slot2].op)) return 0;
    std::swap(b1[slot1].tau, b2[slot2].tau);

    // Catalog entry for an ordered (later, earlier) bilinear pair; -1 if absent, or if the same
    // pair was registered more than once (then a vertex's entry would be ambiguous, so the move
    // rejects in both directions to stay symmetric)
    auto find_entry = [&](op_desc_pair_t const &later, op_desc_pair_t const &earlier) {
      int found = -1;
      for (int i = 0; i < int(data.dyn_op_list.size()); ++i) {
        if (data.dyn_op_list[i].op1 == later && data.dyn_op_list[i].op2 == earlier) {
          if (found != -1) return -1;
          found = i;
        }
      }
      return found;
    };
    if (find_entry(vertex1.ops.op1, vertex1.ops.op2) == -1 || find_entry(vertex2.ops.op1, vertex2.ops.op2) == -1) return 0;

    // Rebuild each vertex in catalog form (op1 at the later time); reject if that pair has no vertex
    auto make_vertex = [&](std::array<timed_bilinear, 2> const &b, configuration::dyn_bosonic_pair_t &vertex) {
      bool first_is_later  = b[0].tau > b[1].tau;
      auto const &later   = first_is_later ? b[0] : b[1];
      auto const &earlier = first_is_later ? b[1] : b[0];
      int entry           = find_entry(later.op, earlier.op);
      if (entry == -1) return false;
      vertex = {data.dyn_op_list[entry], later.tau, earlier.tau};
      return true;
    };
    if (!make_vertex(b1, new_vertex1) || !make_vertex(b2, new_vertex2)) return 0;

    // Same operators at the same times: only the couplings of the two vertices change
    auto coupling = [&](configuration::dyn_bosonic_pair_t const &v) {
      return data.dyn_interactions[v.ops.f_index](double(v.tau1 - v.tau2));
    };
    double old_couplings = coupling(vertex1) * coupling(vertex2);
    if (old_couplings == 0.0) return 0;
    return coupling(new_vertex1) * coupling(new_vertex2) / old_couplings;
  }

  // -------------------------------------------------------------

  mc_weight_t move_swap_dyn::accept() {
    config.dyn_oplist[index1] = new_vertex1;
    config.dyn_oplist[index2] = new_vertex2;
    config.finalize();
    return 1.0; // trace, determinants and permutation sign are unchanged
  }

  // -------------------------------------------------------------

  void move_swap_dyn::reject() { config.finalize(); }

} // namespace triqs_cthyb
