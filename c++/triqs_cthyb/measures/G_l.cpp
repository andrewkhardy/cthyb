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

  measure_G_l::measure_G_l(std::optional<G_l_t> &G_l_opt, qmc_data const &data, int n_l, gf_struct_t const &gf_struct) : data(data), average_sign(0) {
    G_l_opt = block_gf<legendre>{{data.config.beta(), Fermion, n_l}, gf_struct};
    G_l.rebind(*G_l_opt);
    G_l() = 0.0;
  }

  void measure_G_l::accumulate(mc_weight_t s) {
    s *= data.atomic_reweighting;
    average_sign += s;

    double beta = data.config.beta();
    auto Tn     = triqs::utility::legendre_generator();

    for (auto block_idx : range(G_l.size())) {

      foreach (data.dets[block_idx], [this, s, block_idx, beta, &Tn](op_t const &x, op_t const &y, det_scalar_t M) {

        double poly_arg = 2 * double(y.first - x.first) / beta - 1.0;
        Tn.reset(poly_arg);

        auto val = (y.first >= x.first ? s : -s) * M;

        // Apply Lang-Firsov shift to measure the bare Green's function
        val *= lf_shift(block_idx, y, x);

        for (auto l : G_l[block_idx].mesh()) {
          // Evaluate all polynomial orders
          this->G_l[block_idx][l](y.second, x.second) += val * Tn.next();
        }
      })
        ;
    } // for block_idx
  }

  void measure_G_l::collect_results(mpi::communicator const &c) {

    average_sign = mpi::all_reduce(average_sign, c);
    G_l          = mpi::all_reduce(G_l, c);

    double beta = data.config.beta();

    for (auto &G_l_block : G_l) {
      for (auto l : G_l_block.mesh()) {
        /// Normalize polynomial coefficients with basis overlap
        G_l_block[l] *= -(sqrt(2.0 * l.index() + 1.0) / (real(average_sign) * beta));
      }
      matrix<double> id(G_l_block.target_shape());
      id() = 1.0; // this creates an unit matrix
      enforce_discontinuity(G_l_block, id);
    }
  }

  double measure_G_l::lf_shift(long const block, op_t const &y, op_t const &x) {
    if (!data.use_lang_firsov || data.K_n_size == 0) return 1.0;

    double beta = data.config.beta();
    
    // Get the linear orbital indices 'a' and 'b' for y and x
    int a = data.linindex.at({block, static_cast<int>(y.second)});
    int b = data.linindex.at({block, static_cast<int>(x.second)});

    double dtau = double(y.first - x.first);
    if (dtau < 0.0) dtau += beta;
    double poly_arg = 2.0 * dtau / beta - 1.0;
    
    triqs::utility::legendre_generator gen;
    gen.reset(poly_arg);
    
    double K_tau = 0.0;
    for (int n = 0; n < data.K_n_size; ++n) {
      K_tau += data.K_n[a][b][n] * gen.next();
    }

    // Multiply the dressed G(tau) by the analytical bosonic factor B(tau) = exp(-K(tau))
    // to measure the bare G(tau).
    return std::exp(-K_tau);
  }

} // namespace triqs_cthyb
