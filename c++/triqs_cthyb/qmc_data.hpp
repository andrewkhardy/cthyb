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
#include "impurity_trace.hpp"
#include <triqs/gfs.hpp>
#include <triqs/mesh.hpp>
#include <triqs/det_manip.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <span>

namespace triqs_cthyb {
  using namespace triqs::gfs;
  using namespace triqs::mesh;
  using namespace nda;

  // Trace operators with their times
  using timed_op_t  = std::pair<time_pt, op_desc>;
  using timed_ops_t = std::vector<timed_op_t>;

  /************************
 * The Monte Carlo data
 ***********************/
  struct qmc_data {

    configuration config; // Configuration
    time_segment tau_seg;
    std::map<std::pair<int, int>, int> linindex; // Linear index constructed from block and inner indices
    atom_diag const &h_diag;                     // Diagonalization of the atomic problem
    mutable impurity_trace imp_trace;            // Calculator of the trace
    std::vector<int> n_inner;
    block_gf<imtime, delta_target_t> delta; // Hybridization function

    /// This callable object adapts the Delta function for the call of the det.
    struct delta_block_adaptor {
      gf<imtime, delta_target_t> delta_block; // make a copy. Needed in the real case anyway.

      delta_block_adaptor(gf_const_view<imtime, delta_target_t> delta_block) : delta_block(std::move(delta_block)) {}
      delta_block_adaptor(delta_block_adaptor const &)            = default;
      delta_block_adaptor(delta_block_adaptor &&)                 = default;
      delta_block_adaptor &operator=(delta_block_adaptor const &) = delete;
      delta_block_adaptor &operator=(delta_block_adaptor &&)      = default;

      // The difference of two time_pt wraps around beta, the sign does not
      det_scalar_t operator()(std::pair<time_pt, int> const &x, std::pair<time_pt, int> const &y) const {
        det_scalar_t res = delta_block[closest_mesh_pt(double(x.first - y.first))](x.second, y.second);
        return (x.first >= y.first ? res : -res);
      }

      friend void swap(delta_block_adaptor &dba1, delta_block_adaptor &dba2) noexcept { std::swap(dba1.delta_block, dba2.delta_block); }
    };

    std::vector<det_manip::det_manip<delta_block_adaptor>> dets; // The determinants
    int current_sign, old_sign;                                  // Permutation prefactor
    h_scalar_t atomic_weight;                                    // The current value of the trace or norm
    h_scalar_t atomic_reweighting;                               // The current value of the reweighting

    std::vector<bosonic_op_pair_t> dyn_op_list;                  // Catalog of the stochastic dynamical vertex types
    std::vector<std::function<double(double)>> dyn_interactions; // Their couplings D(tau), indexed by f_index

    // Construction
    qmc_data(double beta, solve_parameters_t const &p, atom_diag const &h_diag, std::map<std::pair<int, int>, int> linindex,
             block_gf_const_view<imtime> delta, std::vector<int> n_inner, histo_map_t *histo_map, std::vector<bosonic_op_pair_t> const &dyn_op_list,
             std::vector<std::function<double(double)>> const &dyn_interactions, std::vector<std::vector<std::vector<double>>> const &K_n)
       : config(beta),
         tau_seg(beta),
         linindex(linindex),
         h_diag(h_diag),
         imp_trace(beta, h_diag, histo_map, p.use_norm_as_weight, p.measure_density_matrix, p.performance_analysis),
         n_inner(n_inner),
         delta(map([](gf_const_view<imtime> d) { return real(d); }, delta)),
         current_sign(1),
         old_sign(1),
         dyn_op_list(dyn_op_list),
         dyn_interactions(dyn_interactions) {

      if (!K_n.empty()) build_K_table(K_n, p.verbosity);
      lang_firsov_K_n = K_n;

      std::vector<std::vector<std::pair<time_pt, int>>> X(delta.size()), Y(delta.size());

      // When initial_configuration is given, fill imp_trace and config accordingly
      if (p.initial_configuration) {

        // check that the current beta and beta of the initial configuration are equal
        if (p.initial_configuration->beta() != beta) {
          TRIQS_RUNTIME_ERROR << "Beta of initial configuration not equal current beta: " << p.initial_configuration->beta() << " != " << beta;
        }

        for (auto const &[tau, op] : p.initial_configuration.value()) {
          // check that the block structure is consistent
          if (op.block_index >= delta.size() || op.inner_index >= n_inner[op.block_index]
              || op.linear_index != linindex.at({op.block_index, op.inner_index})) {
            TRIQS_RUNTIME_ERROR << "Inconsistency in the block structure of the initial configuration";
          }

          // insert operators into the impurity trace
          imp_trace.try_insert(tau, op);
          imp_trace.confirm_insert();

          // store tau points and inner block indices for initializing the determinants later
          if (op.dagger)
            X[op.block_index].emplace_back(tau, op.inner_index);
          else
            Y[op.block_index].emplace_back(tau, op.inner_index);

          // insert the operator into the configuration
          config.insert(tau, op);
        }

        for (auto bl : range(delta.size())) {
          if (X[bl].size() != Y[bl].size())
            TRIQS_RUNTIME_ERROR << "Unequal number of c_dag and c operators in bl " << bl << " of the initial configuration";
        }
      }

      // initialize the atomic weight
      std::tie(atomic_weight, atomic_reweighting) = imp_trace.compute();

      // initialize hybridization determinants
      dets.clear();
      dets.reserve(delta.size());
      for (auto const &bl : range(delta.size())) {
#ifdef HYBRIDISATION_IS_COMPLEX
        auto delta_functor = delta_block_adaptor(delta[bl]);
#else
        if (!is_gf_real(delta[bl], 1e-10)) {
          if (p.verbosity >= 2) {
            std::cerr << "WARNING: The Delta(tau) block number " << bl << " is not real in tau space\n";
            std::cerr << "WARNING: max(Im[Delta(tau)]) = " << max_element(abs(imag(delta[bl].data()))) << "\n";
            std::cerr << "WARNING: Disregarding the imaginary component in the calculation.\n";
          }
        }
        auto delta_functor = delta_block_adaptor(real(delta[bl]));
#endif
        if (X[bl].empty()) {
          dets.emplace_back(delta_functor, p.det_init_size);
        } else {
          dets.emplace_back(delta_functor, X[bl], Y[bl]);
        }
        dets.back().set_singular_threshold(p.det_singular_threshold);
        dets.back().set_n_operations_before_check(p.det_n_operations_before_check);
        dets.back().set_precision_warning(p.det_precision_warning);
        dets.back().set_precision_error(p.det_precision_error);
      }
      update_sign();
    }

    qmc_data(qmc_data const &)            = delete; // Member imp_trace is not copyable
    qmc_data &operator=(qmc_data const &) = delete;

    /// The trace operators of a dynamical vertex: op1 as opL(tau1) opR(tau1 - eps), op2 likewise at tau2
    std::array<timed_op_t, 4> dyn_vertex_ops(configuration::dyn_bosonic_pair_t const &v) const {
      auto eps = tau_seg.get_epsilon();
      return {{{v.tau1, v.ops.op1.opL}, {v.tau1 - eps, v.ops.op1.opR}, {v.tau2, v.ops.op2.opL}, {v.tau2 - eps, v.ops.op2.opR}}};
    }

    /// The coupling D(tau1 - tau2) of a dynamical vertex
    double dyn_coupling(configuration::dyn_bosonic_pair_t const &v) const { return dyn_interactions[v.ops.f_index](double(v.tau1 - v.tau2)); }

    /// Catalog index of the vertex type (op1, op2), or -1 if there is none or more than one
    int find_dyn_type(op_desc_pair_t const &op1, op_desc_pair_t const &op2) const {
      int type = -1;
      for (int i = 0; i < int(dyn_op_list.size()); ++i) {
        if (dyn_op_list[i].op1 != op1 || dyn_op_list[i].op2 != op2) continue;
        if (type != -1) return -1;
        type = i;
      }
      return type;
    }

    /// f(tau, op) for every operator in the trace, those of the dynamical vertices included (written out: no allocation)
    template <typename F> void for_each_trace_op(F &&f) const {
      for (auto const &[tau, op] : config) f(tau, op);
      auto eps = tau_seg.get_epsilon();
      for (auto const &v : config.dyn_oplist) {
        f(v.tau1, v.ops.op1.opL);
        f(v.tau1 - eps, v.ops.op1.opR);
        f(v.tau2, v.ops.op2.opL);
        f(v.tau2 - eps, v.ops.op2.opR);
      }
    }

    timed_ops_t trace_ops() const {
      timed_ops_t ops;
      ops.reserve(config.size() + 4 * config.dyn_oplist.size());
      for_each_trace_op([&ops](time_pt const &tau, op_desc const &op) { ops.emplace_back(tau, op); });
      return ops;
    }

    // Lang-Firsov kernel K_ab(tau) on [0, beta/2] (K(tau) = K(beta - tau)): K and its slope on a grid, interpolated by
    // cubic Hermite polynomials, the grid refined until they reproduce the Legendre series to K_TABLE_RTOL. One row per
    // distinct K_ab.
    static constexpr double K_TABLE_RTOL      = 1e-9;
    static constexpr double K_TABLE_WARN_RTOL = 1e-4; // warn only when the table stops this far from the series
    static constexpr long K_TABLE_N_START     = 64;
    static constexpr long K_TABLE_N_MAX       = 1L << 20;

    long n_K_lin       = 0;    // number of linear indices
    long n_K_tab       = 0;    // number of grid intervals
    double K_tab_inv_h = 0.0;  // 1 / grid spacing
    std::vector<long> K_row;   // the row of K_ab at a * n_K_lin + b, -1 where K_ab = 0
    std::vector<double> K_tab; // rows of n_K_tab + 1 points, each K and h K', empty without Lang-Firsov vertices
    std::vector<std::vector<std::vector<double>>> lang_firsov_K_n; // the Legendre coefficients K_n[a][b][n] of the table

    /// Whether occupation kinks at one time, kinks[a] for linear index a (+1 per c^dagger_a, -1 per c_a), leave the
    /// Lang-Firsov weight alone: they interact with a kink of b through sum_a kinks[a] K_ab, which must vanish for every b
    bool lang_firsov_blind_to(std::vector<double> const &kinks) const {
      double scale = 0.0;
      for (auto const &row : lang_firsov_K_n)
        for (auto const &k : row)
          for (double k_n : k) scale = std::max(scale, std::abs(k_n));
      for (size_t b = 0; b < lang_firsov_K_n.size(); ++b)
        for (size_t n = 0; n < lang_firsov_K_n[b][b].size(); ++n) {
          double sum = 0.0;
          for (size_t a = 0; a < lang_firsov_K_n.size(); ++a) sum += kinks[a] * lang_firsov_K_n[a][b][n];
          if (std::abs(sum) > 1e-12 * scale) return false;
        }
      return true;
    }

    // The cubic Hermite interpolant at u in [0, 1] of an interval, from p = {K_i, h K'_i, K_{i+1}, h K'_{i+1}}
    static double hermite(double const *p, double u) {
      double const v = 1.0 - u;
      return v * v * ((1.0 + 2.0 * u) * p[0] + u * p[1]) + u * u * ((3.0 - 2.0 * u) * p[2] - v * p[3]);
    }

    void build_K_table(std::vector<std::vector<std::vector<double>>> const &K_n, int verbosity) {
      double const beta = config.beta(), half = beta / 2.0;

      // K(t) and dK/dt from the Legendre series, with (n + 1) P_{n+1} = (2n + 1) x P_n - n P_{n-1}, P'_{n+1} = P'_{n-1} + (2n + 1) P_n
      auto K_and_slope = [beta](std::vector<double> const &k, double t) {
        double const x = 2.0 * t / beta - 1.0;
        double P = 1.0, P_next = x, dP = 0.0, dP_next = 1.0, K = 0.0, dK = 0.0;
        for (size_t n = 0; n < k.size(); ++n) {
          K += k[n] * P;
          dK += k[n] * dP;
          double const P_after  = ((2.0 * n + 3.0) * x * P_next - (n + 1.0) * P) / (n + 2.0);
          double const dP_after = dP + (2.0 * n + 3.0) * P_next;
          P = P_next, P_next = P_after, dP = dP_next, dP_next = dP_after;
        }
        return std::pair{K, 2.0 / beta * dK};
      };

      // The distinct nonzero K_ab
      n_K_lin = long(K_n.size());
      std::vector<std::vector<double> const *> rows;
      K_row.assign(n_K_lin * n_K_lin, -1);
      for (long a = 0; a < n_K_lin; ++a)
        for (long b = 0; b < n_K_lin; ++b) {
          auto const &k = K_n[a][b];
          if (std::all_of(k.begin(), k.end(), [](double x) { return x == 0.0; })) continue;
          auto same = std::find_if(rows.begin(), rows.end(), [&](auto const *r) { return *r == k; });
          K_row[a * n_K_lin + b] = same - rows.begin();
          if (same == rows.end()) rows.push_back(&k);
        }

      double max_err = 0.0, max_K = 0.0;
      for (n_K_tab = K_TABLE_N_START;; n_K_tab *= 2) {
        double const h = half / double(n_K_tab);
        K_tab.assign(rows.size() * (n_K_tab + 1) * 2, 0.0);
        max_err = max_K = 0.0;
        for (size_t r = 0; r < rows.size(); ++r) {
          double *row = &K_tab[r * (n_K_tab + 1) * 2];
          for (long i = 0; i <= n_K_tab; ++i) {
            auto const [K, slope] = K_and_slope(*rows[r], double(i) * h);
            row[2 * i]            = K;
            row[2 * i + 1]        = h * slope;
            max_K                 = std::max(max_K, std::abs(K));
          }
          for (long i = 0; i < n_K_tab; ++i)
            for (double u : {0.25, 0.5, 0.75})
              max_err = std::max(max_err, std::abs(hermite(row + 2 * i, u) - K_and_slope(*rows[r], (double(i) + u) * h).first));
        }
        if (max_err <= K_TABLE_RTOL * std::max(1.0, max_K) || n_K_tab >= K_TABLE_N_MAX) break;
      }
      K_tab_inv_h = double(n_K_tab) / half;
      if (max_err > K_TABLE_WARN_RTOL * std::max(1.0, max_K))
        std::cerr << "WARNING: Lang-Firsov K(tau) table did not reach its tolerance: max interpolation error " << max_err
                  << " at " << n_K_tab << " intervals (max|K| = " << max_K << ")\n";
      else if (verbosity >= 3)
        std::cout << "Lang-Firsov K(tau) tabulated in " << rows.size() << " distinct row(s) on " << n_K_tab
                  << " intervals over [0, beta/2], max interpolation error " << max_err << " (max|K| = " << max_K << ")" << std::endl;
    }

    /// s_1 s_2 K_{a(op1) b(op2)}(tau1 - tau2) from the table, s = +1 for c^dagger, -1 for c, for dt = tau1 - tau2 in (-beta, beta)
    /// (a double, cheaper than the difference of two time_pt in this inner loop of every move)
    double eval_K(op_desc const &op1, op_desc const &op2, double dt) const {
      long const row = K_row[op1.linear_index * n_K_lin + op2.linear_index];
      if (row < 0) return 0.0;
      double const beta = config.beta();
      double t          = (dt < 0) ? dt + beta : dt; // cyclic difference, in [0, beta)
      if (t > beta / 2.0) t = beta - t;
      double const x   = t * K_tab_inv_h;
      long const i     = std::min(long(x), n_K_tab - 1);
      double const val = hermite(&K_tab[(row * (n_K_tab + 1) + i) * 2], x - double(i));
      return (op1.dagger == op2.dagger) ? val : -val;
    }

    /// Ratio exp(sum s s' K(tau - tau')) of the Lang-Firsov weights for inserting and removing these trace operators
    double compute_lang_firsov_ratio(std::span<timed_op_t const> inserted, std::span<timed_op_t const> removed) const {
      if (K_tab.empty()) return 1.0;
      double delta_W = 0.0;

      // Every moved operator with every trace operator that stays (trace times are unique), in one pass over the trace
      for_each_trace_op([&](time_pt const &tau_bg, op_desc const &op_bg) {
        for (auto const &r : removed)
          if (r.first == tau_bg) return;
        double const t_bg = double(tau_bg);
        for (auto const &[t, op] : inserted) delta_W += eval_K(op, op_bg, double(t) - t_bg);
        for (auto const &[t, op] : removed) delta_W -= eval_K(op, op_bg, double(t) - t_bg);
      });

      // within the inserted and within the removed operators, each pair once (K(0) = 0)
      auto cross = [&](std::span<timed_op_t const> ops, double sign) {
        for (size_t i = 0; i < ops.size(); ++i)
          for (size_t j = i + 1; j < ops.size(); ++j)
            delta_W += sign * eval_K(ops[i].second, ops[j].second, double(ops[i].first) - double(ops[j].first));
      };
      cross(inserted, +1.0);
      cross(removed, -1.0);
      return std::exp(delta_W);
    }

    /// The same for operators listed in place, compute_lang_firsov_ratio({{tau1, op1}, {tau2, op2}}, {})
    double compute_lang_firsov_ratio(std::initializer_list<timed_op_t> inserted, std::initializer_list<timed_op_t> removed) const {
      return compute_lang_firsov_ratio(std::span(inserted.begin(), inserted.size()), std::span(removed.begin(), removed.size()));
    }

    // Occupation kinks at one time, as (linear index a, +1 per c^dagger_a and -1 per c_a)
    using kinks_t = std::vector<std::pair<long, double>>;

    /// The exponent of the Lang-Firsov weight of kinks put at time t in the configuration: their interaction with every
    /// trace operator, sum_a kinks[a] s K_{a b}(t - t') over the operators (t', b, s)
    double lang_firsov_potential(kinks_t const &kinks, double t) const {
      if (K_tab.empty()) return 0.0;
      double phi = 0.0;
      for_each_trace_op([&](time_pt const &tau, op_desc const &op) {
        for (auto const &[a, n] : kinks) phi += n * eval_K(op_desc{0, 0, true, int(a)}, op, t - double(tau));
      });
      return phi;
    }

    /// The exponent of the Lang-Firsov weight between kinks at times dt apart, sum_ab kinks1[a] kinks2[b] K_ab(dt)
    double lang_firsov_interaction(kinks_t const &kinks1, kinks_t const &kinks2, double dt) const {
      if (K_tab.empty()) return 0.0;
      double phi = 0.0;
      for (auto const &[a, n] : kinks1)
        for (auto const &[b, m] : kinks2) phi += n * m * eval_K(op_desc{0, 0, true, int(a)}, op_desc{0, 0, true, int(b)}, dt);
      return phi;
    }

    void update_sign() {

      int s             = 0;
      size_t num_blocks = dets.size();
      std::vector<int> n_op_with_a_equal_to(num_blocks, 0), n_ndag_op_with_a_equal_to(num_blocks, 0);

      // In this first part we compute the sign to bring the configuration to
      // d^_1 d^_1 d^_1 ... d_1 d_1 d_1   d^_2 d^_2 ... d_2 d_2   ...   d^_n .. d_n

      // loop over the operators "op" in the trace (right to left)
      for (auto const &op : config) {

        // how many operators with an 'a' larger than "op" are there on the left of "op"?
        for (int a = op.second.block_index + 1; a < num_blocks; ++a) s += n_op_with_a_equal_to[a];
        n_op_with_a_equal_to[op.second.block_index]++;

        // if "op" is not a dagger how many operators of the same a but with a dagger are there on his right?
        if (op.second.dagger)
          s += n_ndag_op_with_a_equal_to[op.second.block_index];
        else
          n_ndag_op_with_a_equal_to[op.second.block_index]++;
      }

      // Now we compute the sign to bring the configuration to
      // d_1 d^_1 d_1 d^_1 ... d_1 d^_1   ...   d_n d^_n ... d_n d^_n
      for (int block_index = 0; block_index < num_blocks; block_index++) {
        int n = dets[block_index].size();
        s += n * (n + 1) / 2;
      }

      old_sign     = current_sign;
      current_sign = (s % 2 == 0 ? 1 : -1);
    }
  };

  //--------- DEBUG ---------

  using det_type = det_manip::det_manip<qmc_data::delta_block_adaptor>;

  // Print the taus of the c_dag and c of a det
  inline void print_det_sequence(det_type const &det) {
    for (int i = 0; i < det.size(); ++i) std::cout << " ic_dag = " << i << ": tau = " << det.get_x(i).first << std::endl;
    for (int i = 0; i < det.size(); ++i) std::cout << " ic     = " << i << ": tau = " << det.get_y(i).first << std::endl;
  }

  // Check if dets are correctly ordered, otherwise complain
  inline void check_det_sequence(det_type const &det, int config_id) {
    if (det.size() == 0) return;
    auto tau = det.get_x(0).first;
    for (auto ic_dag = 0; ic_dag < det.size(); ++ic_dag) { // c_dag
      if (tau < det.get_x(ic_dag).first) {
        std::cout << "ERROR in det order in config " << config_id << std::endl;
        print_det_sequence(det);
        TRIQS_RUNTIME_ERROR << "Det ordering wrong: tau(ic_dag = " << ic_dag << ") = " << double(det.get_x(ic_dag).first);
      }
      tau = det.get_x(ic_dag).first;
    }
    tau = det.get_y(0).first;
    for (auto ic = 0; ic < det.size(); ++ic) { // c
      if (tau < det.get_y(ic).first) {
        std::cout << "ERROR in det order in config " << config_id << std::endl;
        print_det_sequence(det);
        TRIQS_RUNTIME_ERROR << "Det ordering wrong: tau(ic     = " << ic << ") = " << double(det.get_y(ic).first);
      }
      tau = det.get_y(ic).first;
    }
  }
} // namespace triqs_cthyb