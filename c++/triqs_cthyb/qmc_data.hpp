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
#include <triqs/utility/legendre.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace triqs_cthyb {
  using namespace triqs::gfs;
  using namespace triqs::mesh;
  using namespace nda;

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

      det_scalar_t operator()(std::pair<time_pt, int> const &x, std::pair<time_pt, int> const &y) const {
        det_scalar_t res = delta_block[closest_mesh_pt(double(x.first - y.first))](x.second, y.second);
        return (x.first >= y.first ? res : -res); // x,y first are time_pt, wrapping is automatic in the - operation, but need to
                                                  // compute the sign
      }

      friend void swap(delta_block_adaptor &dba1, delta_block_adaptor &dba2) noexcept { std::swap(dba1.delta_block, dba2.delta_block); }
    };

    std::vector<det_manip::det_manip<delta_block_adaptor>> dets; // The determinants
    int current_sign, old_sign;                                  // Permutation prefactor
    h_scalar_t atomic_weight;                                    // The current value of the trace or norm
    h_scalar_t atomic_reweighting;                               // The current value of the reweighting

    // FIXME : where to put this section of dynamical stuff ?
    std::vector<bosonic_op_pair_t> dyn_op_list;                  // List of bosonic operator pairs for dynamic interactions
    std::vector<std::function<double(double)>> dyn_interactions; // List of dynamic interactions
    //std::vector<gfs::gf<imtime, scalar_valued>> dyn_interactions; // List of dynamic interactions

    // Analytic Density-Density bath support
    // Matrix of k_n polynomials: K_n[a][b][n] where a and b are linear indices.
    std::vector<std::vector<std::vector<double>>> K_n; 
    int K_n_size = 0; // size of polynomials
    bool use_lang_firsov = false;

    // Construction
    qmc_data(double beta, solve_parameters_t const &p, atom_diag const &h_diag, std::map<std::pair<int, int>, int> linindex,
             block_gf_const_view<imtime> delta, std::vector<int> n_inner, histo_map_t *histo_map,
             std::vector<bosonic_op_pair_t> const &dyn_op_list_ = {},
             std::vector<std::function<double(double)>> const &dyn_interactions_ = {},
             std::vector<std::vector<std::vector<double>>> const &K_n_ = {})
       : config(beta),
         tau_seg(beta),
         linindex(linindex),
         h_diag(h_diag),
         imp_trace(beta, h_diag, histo_map, p.use_norm_as_weight, p.measure_density_matrix, p.performance_analysis),
         n_inner(n_inner),
         delta(map([](gf_const_view<imtime> d) { return real(d); }, delta)),
         current_sign(1),
         old_sign(1),
         dyn_op_list(dyn_op_list_),
         dyn_interactions(dyn_interactions_),
         K_n(K_n_) {

      use_lang_firsov = p.lang_firsov;
      if (!K_n.empty() && !K_n[0].empty()) {
        K_n_size = K_n[0][0].size();
        build_K_table(p.verbosity);
      }

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

    /// The four trace operators of a stochastic dynamical vertex, exactly where
    /// insert_dyn places them: op1 as opL(tau1) opR(tau1 - eps), op2 as opL(tau2) opR(tau2 - eps).
    std::vector<std::pair<time_pt, op_desc>> dyn_vertex_ops(bosonic_op_pair_t const &ops, time_pt tau1, time_pt tau2) const {
      auto eps = tau_seg.get_epsilon();
      return {{tau1, ops.op1.opL}, {tau1 - eps, ops.op1.opR}, {tau2, ops.op2.opL}, {tau2 - eps, ops.op2.opR}};
    }

    /// Call f(tau, op) for every fermion operator in the trace: the hybridization operators in
    /// config, plus the operators of the stochastic dynamical vertices (config.dyn_oplist),
    /// which are not in config's oplist but change orbital occupations all the same. These are
    /// the density kinks the Lang-Firsov weight is built from. Visits in place rather than
    /// returning a vector: compute_lang_firsov_ratio runs on every proposal.
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

    /// The same operators as for_each_trace_op, collected into a vector
    std::vector<std::pair<time_pt, op_desc>> trace_ops() const {
      std::vector<std::pair<time_pt, op_desc>> ops;
      ops.reserve(config.size() + 4 * config.dyn_oplist.size());
      for_each_trace_op([&ops](time_pt const &tau, op_desc const &op) { ops.emplace_back(tau, op); });
      return ops;
    }

    /// An operator-free arc of the imaginary-time circle: (lo, lo + length), exclusive
    struct trace_gap_t {
      time_pt lo, length;
      bool full = false; // no operators at all: the whole circle

      bool contains(time_pt const &t) const {
        if (full) return true;
        auto const offset = t - lo;
        return offset > time_pt{} && offset < length; // time_pt compares grid positions only
      }
    };

    /// The operator-free arc of the trace containing tau, bounded by the nearest trace operators
    /// below and above it (cyclically), with the operators of dyn_oplist[skip] left out. This is
    /// what move_insert_dyn's local proposal draws into; see moves/insert_dyn.cpp.
    trace_gap_t trace_gap(time_pt const &tau, long skip = -1) const {
      bool found = false;
      time_pt up, down; // distance to the nearest operator above / below tau
      auto visit = [&](time_pt const &t) {
        if (t == tau) return;
        auto const d_up = t - tau, d_down = tau - t;
        if (!found || d_up < up) up = d_up;
        if (!found || d_down < down) down = d_down;
        found = true;
      };
      auto const eps     = tau_seg.get_epsilon();
      auto visit_vertex  = [&](configuration::dyn_bosonic_pair_t const &v) {
        visit(v.tau1);
        visit(v.tau1 - eps);
        visit(v.tau2);
        visit(v.tau2 - eps);
      };
      for (auto const &[t, op] : config) visit(t);
      for (long k = 0; k < long(config.dyn_oplist.size()); ++k)
        if (k != skip) visit_vertex(config.dyn_oplist[k]);

      if (!found) return {tau, tau_seg.get_upper_pt(), true};
      // One operator time only (cannot happen for a valid trace, which has operators in pairs):
      // up + down is the whole circle, which time_pt's cyclic addition would wrap to zero
      auto length = up + down;
      if (length == tau_seg.get_lower_pt()) length = tau_seg.get_upper_pt();
      return {tau - down, length, false};
    }

    /// Probability density with which move_insert_dyn proposes a vertex at (tau1 > tau2) of a
    /// given catalog entry, into the trace with dyn_oplist[skip] left out (skip = -1: as is).
    /// A mixture: with probability p_local both ends in one operator-free arc of length l
    /// (density 2 / (beta l), zero if the pair does not share an arc), otherwise both uniform on
    /// [0, beta) (density 2 / beta^2). move_remove_dyn uses the same function for the reverse.
    double dyn_insertion_density(time_pt const &tau1, time_pt const &tau2, double p_local, long skip = -1) const {
      double const beta = config.beta();
      double density    = (1.0 - p_local) * (2.0 / (beta * beta));
      if (p_local > 0.0) {
        auto const gap = trace_gap(tau1, skip);
        if (gap.contains(tau2)) density += p_local * 2.0 / (beta * double(gap.length));
      }
      return density * (1.0 / dyn_op_list.size());
    }

    // ---------------------------------------------------------------------------------
    // Tabulated Lang-Firsov kernel
    //
    // K_{ab}(tau) is needed for every (proposed operator, trace operator) pair of every
    // proposal -- ~4 x 100 evaluations per move at beta = 100 -- so summing the dyn_n_l
    // Legendre terms each time dominated the move cost. It is instead tabulated once on a
    // uniform grid over [0, beta/2] (K(tau) = K(beta - tau)) and linearly interpolated. The
    // grid is doubled until interpolation reproduces the Legendre series to
    // K_TABLE_RTOL * max(1, max|K|) at every midpoint, so the table is a faithful stand-in for
    // the series rather than a further approximation of the model: the series itself differs
    // from the exact double integral of D(tau) by ~1e-5 at dyn_n_l = 50. The MC stays exact
    // for the tabulated K -- every weight is computed from the same table.
    // ---------------------------------------------------------------------------------
    static constexpr double K_TABLE_RTOL  = 1e-9;
    static constexpr long K_TABLE_N_START = 4096;
    static constexpr long K_TABLE_N_MAX   = 1L << 20;

    long n_K_lin = 0;          // K_tab is indexed by (linear index a, linear index b, grid point)
    long n_K_tab = 0;          // number of grid intervals over [0, beta/2]
    double K_tab_inv_h = 0.0;  // 1 / grid spacing
    std::vector<double> K_tab; // (n_K_lin * n_K_lin) rows of n_K_tab + 1 points

    /// K_{ab}(t) summed from its Legendre coefficients -- the reference the table is built from
    double K_legendre(long a, long b, double t) const {
      double const beta = config.beta();
      triqs::utility::legendre_generator leg;
      leg.reset(2.0 * t / beta - 1.0);
      double val = 0.0;
      for (int n = 0; n < K_n_size; ++n) val += K_n[a][b][n] * leg.next();
      return val;
    }

    void build_K_table(int verbosity) {
      n_K_lin           = long(K_n.size());
      double const half = config.beta() / 2.0;
      double max_err = 0.0, max_K = 0.0;
      for (n_K_tab = K_TABLE_N_START;; n_K_tab *= 2) {
        double const h = half / double(n_K_tab);
        K_tab.assign(n_K_lin * n_K_lin * (n_K_tab + 1), 0.0);
        max_err = max_K = 0.0;
        for (long a = 0; a < n_K_lin; ++a)
          for (long b = 0; b < n_K_lin; ++b) {
            if (std::all_of(K_n[a][b].begin(), K_n[a][b].end(), [](double k) { return k == 0.0; })) continue;
            double *row = &K_tab[(a * n_K_lin + b) * (n_K_tab + 1)];
            for (long i = 0; i <= n_K_tab; ++i) {
              row[i] = K_legendre(a, b, double(i) * h);
              max_K  = std::max(max_K, std::abs(row[i]));
            }
            for (long i = 0; i < n_K_tab; ++i)
              max_err = std::max(max_err, std::abs(0.5 * (row[i] + row[i + 1]) - K_legendre(a, b, (double(i) + 0.5) * h)));
          }
        if (max_err <= K_TABLE_RTOL * std::max(1.0, max_K) || n_K_tab >= K_TABLE_N_MAX) break;
      }
      K_tab_inv_h = double(n_K_tab) / half;
      if (max_err > K_TABLE_RTOL * std::max(1.0, max_K))
        std::cerr << "WARNING: Lang-Firsov K(tau) table did not reach its tolerance: max interpolation error " << max_err
                  << " at " << n_K_tab << " intervals (max|K| = " << max_K << ")\n";
      else if (verbosity >= 3)
        std::cout << "Lang-Firsov K(tau) tabulated on " << n_K_tab << " intervals over [0, beta/2], max interpolation error "
                  << max_err << " (max|K| = " << max_K << ")" << std::endl;
    }

    /// s_1 s_2 K_{a(op1) b(op2)}(tau1 - tau2) from the table, s = +1 for c^dagger, -1 for c
    double eval_K(op_desc const &op1, op_desc const &op2, time_pt const &tau1, time_pt const &tau2) const {
      double const beta = config.beta();
      double t          = double(tau1 - tau2); // cyclic difference, in [0, beta)
      if (t > beta / 2.0) t = beta - t;
      double const x    = t * K_tab_inv_h;
      long const i      = std::min(long(x), n_K_tab - 1);
      double const *row = &K_tab[(op1.linear_index * n_K_lin + op2.linear_index) * (n_K_tab + 1)];
      double const val  = row[i] + (x - double(i)) * (row[i + 1] - row[i]);
      return (op1.dagger == op2.dagger) ? val : -val;
    }

/// Ratio of dynamical MC weights w^dyn_loc for a proposed operator update:
///   exp{ Σ_{op pairs (α,β)} s_α s_β K_{i(α)j(β)}(τ_α - τ_β) }
double compute_lang_firsov_ratio(
    std::vector<std::pair<time_pt, op_desc>> const& inserted,
    std::vector<std::pair<time_pt, op_desc>> const& removed) const {

  if (!use_lang_firsov || K_n_size == 0) return 1.0;

#ifdef CTHYB_DEBUG
  // eval_K indexes the table by op.linear_index, so an op_desc rebuilt by a move with that field
  // left at a default silently reads another orbital's kernel (remove.cpp once did exactly this)
  for (auto const *ops : {&inserted, &removed})
    for (auto const &[t, op] : *ops)
      if (op.linear_index != linindex.at({op.block_index, op.inner_index}))
        TRIQS_RUNTIME_ERROR << "compute_lang_firsov_ratio: " << op << " carries linear_index " << op.linear_index;
#endif

  double delta_W = 0.0;

  // 1. Interactions with the persistent background: every trace operator not being removed,
  //    hybridization and stochastic dynamical-vertex operators alike
  auto background_interaction = [&](op_desc const& op, time_pt const& t, double sign) {
    for_each_trace_op([&](time_pt const& t_bg, op_desc const& op_bg) {
      bool is_removed = std::any_of(removed.begin(), removed.end(), [&](auto const& r) {
        return r.first == t_bg && r.second.block_index == op_bg.block_index && r.second.inner_index == op_bg.inner_index
           && r.second.dagger == op_bg.dagger;
      });
      if (!is_removed) delta_W += sign * eval_K(op, op_bg, t, t_bg);
    });
  };
  for (auto const& [t, op] : inserted) background_interaction(op, t, +1.0);
  for (auto const& [t, op] : removed)  background_interaction(op, t, -1.0);

  // 2. Cross-interactions within inserted/removed sets (each pair once, i < j)
  // Note: diagonal terms K(0) = 0 by the Dirichlet boundary condition.
  auto cross = [&](auto const& ops, double sign) {
    for (size_t i = 0; i < ops.size(); ++i)
      for (size_t j = i + 1; j < ops.size(); ++j)
        delta_W += sign * eval_K(ops[i].second, ops[j].second, ops[i].first, ops[j].first);
  };
  cross(inserted, +1.0);
  cross(removed,  -1);

  return std::exp(delta_W);
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

  // Print taus of operator sequence in dets
  inline void print_det_sequence(qmc_data const &data) {
    int i;
    int block_index;
    for (block_index = 0; block_index < data.dets.size(); ++block_index) {
      auto det = data.dets[block_index];
      if (det.size() == 0) return;
      std::cout << "BLOCK = " << block_index << std::endl;
      for (i = 0; i < det.size(); ++i) { // c_dag
        std::cout << " ic_dag = " << i << ": tau = " << det.get_x(i).first << std::endl;
      }
      for (i = 0; i < det.size(); ++i) { // c
        std::cout << " ic     = " << i << ": tau = " << det.get_y(i).first << std::endl;
      }
    }
  }

  inline void print_det_sequence(det_type const &det) {
    int i;
    for (i = 0; i < det.size(); ++i) { // c_dag
      std::cout << " ic_dag = " << i << ": tau = " << det.get_x(i).first << std::endl;
    }
    for (i = 0; i < det.size(); ++i) { // c
      std::cout << " ic     = " << i << ": tau = " << det.get_y(i).first << std::endl;
    }
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