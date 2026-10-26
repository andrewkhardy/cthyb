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
#include "./density_matrix.hpp"
#include <mpi/vector.hpp>

#include <iomanip>
#include <sstream>

namespace triqs_cthyb {

  measure_density_matrix::measure_density_matrix(qmc_data const &data, std::vector<matrix_t> &density_matrix) : data(data), block_dm(density_matrix) {
    block_dm.resize(data.imp_trace.get_density_matrix().size());
    for (int i = 0; i < block_dm.size(); ++i) {
      block_dm[i]   = data.imp_trace.get_density_matrix()[i].mat;
      block_dm[i]() = 0;
    }
  }
  // --------------------

  void measure_density_matrix::accumulate(mc_weight_t s) {
    // we assume here that we are in "Norm" mode, i.e. qmc weight is norm, not trace

    // Recompute without any Yee threshold: the density_matrix in the trace changes at each computation,
    // in particular at the last failed attempt.
    auto const [weight, reweighting] = data.imp_trace.compute();
    // The only source of Tr rho - 1: the trace recomputed here against the stored one, which z gets
    auto const d = (weight * reweighting - data.atomic_weight * data.atomic_reweighting) / data.atomic_weight;
    ++n_measures;
    if (std::abs(d) > 1.e-10) ++n_trace_off;
    trace_dev += s * d;
    z += s * data.atomic_reweighting;
    s /= data.atomic_weight; // accumulate matrix / norm since weight is norm * det

    // Careful: there is no reweighting factor here!
    int size = block_dm.size();
    for (int i = 0; i < size; ++i)
      if (data.imp_trace.get_density_matrix()[i].is_valid) { block_dm[i] += s * data.imp_trace.get_density_matrix()[i].mat; }
  }

  // ---------------------------------------------

  void measure_density_matrix::collect_results(mpi::communicator const &c) {

    z                          = mpi::all_reduce(z, c);
    block_dm                   = mpi::all_reduce(block_dm, c);
    trace_dev                  = mpi::all_reduce(trace_dev, c);
    long const n_total = mpi::all_reduce(n_measures, c), n_off = mpi::all_reduce(n_trace_off, c);
    for (auto &b : block_dm){
        // Normalize
        b /= real(z);

        // Enforce hermiticity
        b = make_regular(0.5*(b + dagger(b)));
    }

    // Check: the trace of the density matrix must be 1 by construction. Every rank takes rank 0's verdict, so a bad
    // trace stops all of them; raising on rank 0 alone left the others waiting in the next collective until the
    // allocation ran out
    h_scalar_t tr = 0;
    for (auto &b : block_dm) tr += trace(b);
    mpi::broadcast(tr, c);
    std::ostringstream why;
    why << " (" << n_off << " of " << n_total << " measurements recomputed a trace that differs from the stored Monte Carlo trace by more "
        << "than 1e-10 of the weight; together they account for Tr - 1 = " << real(trace_dev / z) << ")";
    if (std::abs(tr - 1) > 0.0001) TRIQS_RUNTIME_ERROR << "Trace of the density matrix is " << tr << " instead of 1" << why.str();

    if (c.rank() != 0) return;
    if (std::abs(tr - 1) > 1.e-10)
      std::cerr << "Warning :: Trace of the density matrix is " << std::setprecision(13) << tr << std::setprecision(6) << " instead of 1"
                << why.str() << std::endl;

  }
}
