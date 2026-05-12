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

#include "./G_tau.hpp"

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  measure_G_tau::measure_G_tau(qmc_data const &data, int n_tau, gf_struct_t const &gf_struct, container_set_t &results)
     : data(data), average_sign(0) {
    results.G_tau_accum = block_gf<imtime, G_target_t>({data.config.beta(), Fermion, n_tau}, gf_struct);
    G_tau.rebind(*results.G_tau_accum);
    G_tau() = 0.0;

    results.asymmetry_G_tau = block_gf{G_tau};
    asymmetry_G_tau.rebind(*results.asymmetry_G_tau);
  }

  void measure_G_tau::accumulate(mc_weight_t s) {
    s *= data.atomic_reweighting;
    average_sign += s;

    for (auto block_idx : range(G_tau.size())) {
      foreach (data.dets[block_idx], [this, s, block_idx](op_t const &x, op_t const &y, det_scalar_t M) {
        // beta-periodicity is implicit in the argument, just fix the sign properly
        auto val    = (y.first >= x.first ? s : -s) * M;
        double dtau = double(y.first - x.first);
        this->G_tau[block_idx][closest_mesh_pt(dtau)](y.second, x.second) += val;
      })
        ;
    }
  }
void measure_G_tau::collect_results(mpi::communicator const &c) {

    G_tau        = mpi::all_reduce(G_tau, c);
    average_sign = mpi::all_reduce(average_sign, c);

    for (int bl_idx = 0; bl_idx < G_tau.size(); ++bl_idx) {
      auto &G_tau_block = G_tau[bl_idx];
      double beta = G_tau_block.mesh().beta();
      G_tau_block /= -real(average_sign) * beta * G_tau_block.mesh().delta();

      if (data.use_lang_firsov && data.K_n_size > 0) {
        for (int i = 0; i < G_tau_block.data().shape()[1]; ++i) {
          for (int j = 0; j < G_tau_block.data().shape()[2]; ++j) {
            int a = data.linindex.at({bl_idx, i});
            int b = data.linindex.at({bl_idx, j});

            double K_ii_0 = 0.0;
            double K_jj_0 = 0.0;
            triqs::utility::legendre_generator gen_0;
            gen_0.reset(-1.0);
            for (int n = 0; n < data.K_n_size; ++n) {
              double P_n = gen_0.next();
              K_ii_0 += data.K_n[a][a][n] * P_n;
              K_jj_0 += data.K_n[b][b][n] * P_n;
            }
            double K_0_term = 0.5 * (K_ii_0 + K_jj_0);

            for (auto tau_pt : G_tau_block.mesh()) {
              double tau = double(tau_pt);
              double x = 2.0 * tau / beta - 1.0;
              double K_tau = 0.0;
              
              triqs::utility::legendre_generator gen_tau;
              gen_tau.reset(x);
              for (int n = 0; n < data.K_n_size; ++n) {
                K_tau += data.K_n[a][b][n] * gen_tau.next();
              }
              
              G_tau_block[tau_pt](i, j) *= std::exp(K_tau - K_0_term);
            }
          }
        }
      }

      int last = G_tau_block.mesh().size() - 1;
      G_tau_block[0] *= 2;
      G_tau_block[last] *= 2;

      G_tau_block[0] = 0.5 * matrix_t(G_tau_block[0] - 1 - G_tau_block[last]);
      G_tau_block[last] = -1 - G_tau_block[0];
    }

    asymmetry_G_tau = make_hermitian(G_tau) - G_tau;
    G_tau           = G_tau + asymmetry_G_tau;
  }