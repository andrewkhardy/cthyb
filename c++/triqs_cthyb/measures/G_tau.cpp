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
#include <triqs/utility/itertools.hpp>

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

    for (auto [block_idx, det] : itertools::enumerate(data.dets)) {
      long N = det.size();
      for (long id_y : range(N)) {
        auto y = det.get_y(id_y);
        
        for (long id_x : range(N)) {
          auto x    = det.get_x(id_x);
          auto Minv = det.inverse_matrix(id_y, id_x);
          
          // beta-periodicity is implicit in the argument, just fix the sign properly
          auto val    = (y.first >= x.first ? s : -s) * Minv;
          double dtau = double(y.first - x.first);
          
          // Apply Lang-Firsov shift
          val *= lf_shift(block_idx, y, x);
          
          this->G_tau[block_idx][closest_mesh_pt(dtau)](y.second, x.second) += val;
        }
      }
    }
  }

  void measure_G_tau::collect_results(mpi::communicator const &c) {

    G_tau        = mpi::all_reduce(G_tau, c);
    average_sign = mpi::all_reduce(average_sign, c);

    for (auto &G_tau_block : G_tau) {
      double beta = G_tau_block.mesh().beta();
      G_tau_block /= -real(average_sign) * beta * G_tau_block.mesh().delta();

      // Multiply first and last bins by 2 to account for full bins
      int last = G_tau_block.mesh().size() - 1;
      G_tau_block[0] *= 2;
      G_tau_block[last] *= 2;

      // Enforce discontinuity in Green function
      G_tau_block[0] = 0.5 * matrix_t(G_tau_block[0] - 1 - G_tau_block[last]);
      G_tau_block[last] = -1 - G_tau_block[0];
    }

    // We enforce the fundamental Green function property G(tau)[i,j] = G(tau)*[j,i]
    // and store the symmetry violation separately
    asymmetry_G_tau = make_hermitian(G_tau) - G_tau;
    G_tau           = G_tau + asymmetry_G_tau;
  }

  double measure_G_tau::lf_shift(long const block, op_t const &y, op_t const &x) {
    if (!data.use_lang_firsov || data.K_n_size == 0) return 1.0;

    double I_tau = 0.0;
    double beta = data.config.beta();
    
    // Get the linear orbital index 'a' for the y (annihilation) operator
    int a = data.linindex.at({block, static_cast<int>(y.second)});

    for (auto const &[t_bg, op_bg] : data.config) {
      int b = data.linindex.at({op_bg.block_index, op_bg.inner_index});
      
      // S_bg is +1 for c^\dagger and -1 for c (boundaries of the density segments)
      int S_bg = op_bg.dagger ? 1 : -1;
      
      double t_diff = double(y.first - t_bg);
      if (t_diff < 0.0) t_diff += beta;
      double poly_arg = 2.0 * t_diff / beta - 1.0;
      
      triqs::utility::legendre_generator gen_bg;
      gen_bg.reset(poly_arg);
      
      for (int n = 0; n < data.K_n_size; ++n) {
        double P_n = gen_bg.next();
        // The factor of 2.0 mirrors the cross-term logic in compute_lang_firsov_ratio
        I_tau += data.K_n[a][b][n] * 2.0 * S_bg * P_n;
      }
    }

    // Note: If you are measuring the improved estimator F(tau) (like fprefactor in CTSEG), 
    // you should return I_tau. If you are reweighting to measure the bare G(tau), 
    // you may want to return std::exp(I_tau).
    return I_tau;
  }

} // namespace triqs_cthyb
