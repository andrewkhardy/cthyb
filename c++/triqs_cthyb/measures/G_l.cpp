/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2014, H. U.R. Strand, P. Seth, I. Krivenko, M. Ferrero and O. Parcollet
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

#include "G_l.hpp"

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  measure_G_l::measure_G_l(std::optional<G_l_t> &G_l_opt, qmc_data const &data, int n_l, gf_struct_t const &gf_struct)
     : data(data), average_sign(0), moments(n_l) {
    G_l_opt = block_gf<legendre>{{data.config.beta(), Fermion, n_l}, gf_struct};
    G_l.rebind(*G_l_opt);
    G_l() = 0.0;
    for (auto const &[bl, size] : gf_struct) {
      accumulated.push_back(nda::zeros<mc_weight_t>(size, size, n_l));
      pair_x.emplace_back(size * size);
      pair_w.emplace_back(size * size);
    }
  }

  void measure_G_l::accumulate(mc_weight_t s) {
    s *= data.atomic_reweighting;
    average_sign += s;

    double const beta = data.config.beta();

    for (auto block_idx : range(G_l.size())) {

      // Every element of M, by its inner indices (i, j) = (y, x) with the cyclic dt from x to y
      long const n = accumulated[block_idx].shape()[0];
      for (auto &x : pair_x[block_idx]) x.clear();
      for (auto &w : pair_w[block_idx]) w.clear();
      foreach (data.dets[block_idx], [&](op_t const &x, op_t const &y, det_scalar_t M) {
        double dt       = double(y.first) - double(x.first);
        mc_weight_t val = s * M;
        if (dt < 0) dt += beta, val = -val;
        pair_x[block_idx][y.second * n + x.second].push_back(2 * dt / beta - 1.0);
        pair_w[block_idx][y.second * n + x.second].push_back(val);
      });

      for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j) moments.add(pair_x[block_idx][i * n + j], pair_w[block_idx][i * n + j], &accumulated[block_idx](i, j, 0));
    }
  }

  void measure_G_l::collect_results(mpi::communicator const &c) {

    for (auto block_idx : range(G_l.size()))
      for (auto l : G_l[block_idx].mesh())
        for (long i = 0; i < accumulated[block_idx].shape()[0]; ++i)
          for (long j = 0; j < accumulated[block_idx].shape()[1]; ++j) G_l[block_idx][l](i, j) = accumulated[block_idx](i, j, l.index());

    average_sign = mpi::all_reduce(average_sign, c);
    G_l          = mpi::all_reduce(G_l, c);

    double beta = data.config.beta();

    for (auto &G_l_block : G_l) {
      // Normalize the polynomial coefficients with the basis overlap
      for (auto l : G_l_block.mesh()) G_l_block[l] *= -(sqrt(2.0 * l.index() + 1.0) / (real(average_sign) * beta));
      matrix<double> id(G_l_block.target_shape());
      id() = 1.0; // a scalar assigned to a matrix sets the identity
      enforce_discontinuity(G_l_block, id);
    }
  }

} // namespace triqs_cthyb
