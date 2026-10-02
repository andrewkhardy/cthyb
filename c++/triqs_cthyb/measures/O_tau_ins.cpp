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

#include <triqs/mc_tools.hpp>

#include "./O_tau_ins.hpp"

namespace triqs_cthyb {

  using namespace triqs::gfs;
  using namespace triqs::mesh;

  measure_O_tau_ins::measure_O_tau_ins(std::optional<gf<imtime, scalar_valued>> &O_tau_opt, qmc_data const &data, int n_tau,
                                       many_body_op_t const &op1, many_body_op_t const &op2, int min_ins, mc_tools::random_generator &rng)
    : data(data),
      average_sign(0),
      op1(op1),
      op2(op2),
      min_ins(min_ins),
      rng(rng),
      sweep(min_ins < 0),
      op1_mat(data.h_diag.get_op_mat(op1)),
      op2_mat(data.h_diag.get_op_mat(op2)) {
    O_tau_opt = gf<imtime, scalar_valued>{{data.config.beta(), Boson, n_tau}};
    O_tau.rebind(*O_tau_opt);
    O_tau() = 0.0;

    op1_d = data.imp_trace.attach_aux_operator(op1);
    op2_d = data.imp_trace.attach_aux_operator(op2);

    double const dtau = data.config.beta() / double(n_tau - 1);
    mesh_decay.resize(data.h_diag.n_subspaces());
    for (int b = 0; b < data.h_diag.n_subspaces(); ++b)
      for (int i = 0; i < data.h_diag.get_subspace_dim(b); ++i) mesh_decay[b].push_back(std::exp(-dtau * data.h_diag.get_eigenvalue(b, i)));
  }

  void measure_O_tau_ins::accumulate(mc_weight_t s) {
    if (sweep) {
      accumulate_sweep(s);
      return;
    }

    s *= data.atomic_reweighting;
    average_sign += s;

    // Insertions per measurement: linear in the perturbation order, with min_ins as a floor. Each
    // insertion is an unbiased sample for this configuration, so the count only sets how finely
    // one configuration is probed; consecutive configurations are strongly correlated, so beyond
    // a few dozen insertions the error hardly moves. The former pto * pto was ~1600 insertions
    // (~4 ms, as much as ~1200 moves) per measurement at beta = 100 and half the run time.
    int pto = 0;
    for (const auto &det : data.dets) pto += det.size();
    int nsamples = pto;
    if( nsamples < min_ins ) nsamples = min_ins;

    mc_weight_t atomic_weight, atomic_reweighting;
    auto [bare_atomic_weight, bare_atomic_reweighting] = data.imp_trace.compute();
    const auto prefactor = s / bare_atomic_weight / bare_atomic_reweighting / double(nsamples);

    for (int i : range(nsamples)) {
      auto tau1 = data.tau_seg.get_random_pt(rng);
      auto tau2 = data.tau_seg.get_random_pt(rng);
      double dtau = double(tau2 - tau1);

      try {
        data.imp_trace.try_insert(tau1, op1_d);
        data.imp_trace.try_insert(tau2, op2_d);
        std::tie(atomic_weight, atomic_reweighting) = data.imp_trace.compute();
      } catch (rbt_insert_error const &) { atomic_weight = 0.; }

      data.imp_trace.cancel_insert();
      O_tau[closest_mesh_pt(dtau)] += prefactor * atomic_weight * atomic_reweighting;
    }
  }

  // <O2(tau) O1(0)> of this configuration, exactly, at every mesh point, with O1 at the trace boundary tau = 0
  // (translation invariance). Let R_k be the product of the trace from 0 to t_k, operator k included, and L_k the
  // product from t_{k+1}, operator k + 1 included, to beta. Then O2 at tau in (t_k, t_{k+1}) gives
  //   Tr[L_k e^{-(t_{k+1} - tau) H} O2 e^{-(tau - t_k) H} R_k O1] = sum_mn e^{-(t_{k+1} - tau) E_m} O2_mn e^{-(tau - t_k) E_n} Y_nm
  // with Y = R_k O1 L_k, block by block of h_diag. This is divided by the trace without O1 and O2.
  void measure_O_tau_ins::accumulate_sweep(mc_weight_t s) {
    s *= data.atomic_reweighting;
    average_sign += s;

    auto const &h_diag = data.h_diag;
    double const beta  = data.config.beta();
    auto const _       = nda::range::all;

    // The trace operators in increasing time, those of the dynamical vertices included. Interval k is (t[k], t[k + 1]),
    // and operator k sits at t[k + 1].
    auto ops = data.trace_ops();
    std::sort(ops.begin(), ops.end(), [](auto const &x, auto const &y) { return x.first < y.first; });
    int const n_ops = ops.size();
    std::vector<double> t(n_ops + 2);
    t[0] = 0.0;
    for (int k = 0; k < n_ops; ++k) t[k + 1] = double(ops[k].first);
    t[n_ops + 1] = beta;

    auto op_target = [&](int k, int b) -> long {
      auto const &op = ops[k].second;
      return op.dagger ? h_diag.cdag_connection(op.linear_index, b) : h_diag.c_connection(op.linear_index, b);
    };
    auto op_matrix = [&](int k, int b) -> matrix_t const & {
      auto const &op = ops[k].second;
      return op.dagger ? h_diag.cdag_matrix(op.linear_index, b) : h_diag.c_matrix(op.linear_index, b);
    };

    long const n_mesh = O_tau.mesh().size();
    double const dtau = beta / double(n_mesh - 1);
    std::vector<h_scalar_t> numerator(n_mesh, 0.0), partial_sums;
    h_scalar_t bare_trace = 0.0;
    double trace_abs      = 0.0;

    std::vector<int> block(n_ops + 1);
    std::vector<matrix_t> R(n_ops + 1), L(n_ops + 1);

    for (int b0 = 0; b0 < h_diag.n_subspaces(); ++b0) {

      // The block in each interval, for the trace that starts in b0; it contributes only if it returns to b0
      block[0]    = b0;
      bool broken = false;
      for (int k = 0; k < n_ops && !broken; ++k) {
        block[k + 1] = op_target(k, block[k]);
        broken       = block[k + 1] < 0;
      }
      if (broken || block[n_ops] != b0) continue;

      // e^{-(t[k + 1] - t[k]) E_i} on the block of interval k
      auto decay = [&](int k, int i) { return std::exp(-(t[k + 1] - t[k]) * h_diag.get_eigenvalue(block[k], i)); };

      R[0] = nda::eye<h_scalar_t>(h_diag.get_subspace_dim(b0));
      for (int k = 0; k < n_ops; ++k) {
        matrix_t evolved = R[k];
        for (int i = 0; i < evolved.shape()[0]; ++i) evolved(i, _) *= decay(k, i);
        R[k + 1] = op_matrix(k, block[k]) * evolved;
      }
      L[n_ops] = nda::eye<h_scalar_t>(h_diag.get_subspace_dim(b0));
      for (int k = n_ops - 1; k >= 0; --k) {
        matrix_t evolved = L[k + 1];
        for (int i = 0; i < evolved.shape()[1]; ++i) evolved(_, i) *= decay(k + 1, i);
        L[k] = evolved * op_matrix(k, block[k]);
      }

      h_scalar_t block_trace = 0.0;
      for (int u = 0; u < h_diag.get_subspace_dim(b0); ++u) block_trace += decay(n_ops, u) * R[n_ops](u, u);
      bare_trace += block_trace;
      trace_abs += std::abs(block_trace);

      // O1 and O2 are diagonal in the occupation basis: each maps a block to itself or annihilates it
      if (op1_mat.connection(b0) != b0) continue;
      auto const &O1 = op1_mat.block_mat[b0];

      long j = 0; // first mesh point not yet assigned to an interval
      for (int k = 0; k <= n_ops; ++k) {
        long j_end = j;
        while (j_end < n_mesh && (k == n_ops || double(j_end) * dtau < t[k + 1])) ++j_end;
        long const n_pts = j_end - j;
        int const bk     = block[k];
        if (n_pts == 0 || op2_mat.connection(bk) != bk) {
          j = j_end;
          continue;
        }

        auto const &O2 = op2_mat.block_mat[bk];
        matrix_t Y     = R[k] * O1 * L[k];
        int const d    = h_diag.get_subspace_dim(bk);
        matrix_t A(d, d); // A_mn = O2_mn Y_nm
        for (int m = 0; m < d; ++m)
          for (int n = 0; n < d; ++n) A(m, n) = O2(m, n) * Y(n, m);

        // Forward in tau: partial_m(tau) = sum_n A_mn e^{-(tau - t_k) E_n}. Backward: sum_m e^{-(t_{k+1} - tau) E_m}
        // partial_m(tau). Each factor only decays along its own direction, so the recursions are stable.
        partial_sums.assign(n_pts * d, 0.0);
        std::vector<double> factor(d);
        for (int n = 0; n < d; ++n) factor[n] = std::exp(-(double(j) * dtau - t[k]) * h_diag.get_eigenvalue(bk, n));
        for (long p = 0; p < n_pts; ++p) {
          for (int m = 0; m < d; ++m)
            for (int n = 0; n < d; ++n) partial_sums[p * d + m] += A(m, n) * factor[n];
          for (int n = 0; n < d; ++n) factor[n] *= mesh_decay[bk][n];
        }
        for (int m = 0; m < d; ++m) factor[m] = std::exp(-(t[k + 1] - double(j_end - 1) * dtau) * h_diag.get_eigenvalue(bk, m));
        for (long p = n_pts - 1; p >= 0; --p) {
          for (int m = 0; m < d; ++m) numerator[j + p] += factor[m] * partial_sums[p * d + m];
          for (int m = 0; m < d; ++m) factor[m] *= mesh_decay[bk][m];
        }
        j = j_end;
      }
    }

    // The same trace the Monte Carlo weight holds, up to its truncation of negligible blocks
    h_scalar_t const mc_trace = data.atomic_weight * data.atomic_reweighting;
    if (std::abs(bare_trace - mc_trace) > 1.e-8 * trace_abs)
      TRIQS_RUNTIME_ERROR << "measure_O_tau (sweep): the trace " << bare_trace << " of configuration " << data.config.get_id()
                          << " differs from the Monte Carlo trace " << mc_trace;

    for (auto const &tau : O_tau.mesh()) O_tau[tau] += s * numerator[tau.index()] / bare_trace;
  }

  void measure_O_tau_ins::collect_results(mpi::communicator const &c) {
    O_tau        = mpi::all_reduce(O_tau, c);
    average_sign = mpi::all_reduce(average_sign, c);

    // Every mesh point is evaluated exactly: no bin width, no half-width edge bins
    if (sweep) {
      O_tau *= 1.0 / real(average_sign);
      return;
    }

    O_tau *= double(O_tau.mesh().size() - 1) / real(average_sign);

    // Assuming commuting operators so that O(0) = O(beta)
    // use that to counter the half-sized edge bind, by taking the average
    // on the boundary

    int last     = O_tau.mesh().size() - 1;
    auto average = O_tau[0] + O_tau[last];

    O_tau[0]    = average;
    O_tau[last] = average;
  }
} // namespace triqs_cthyb
