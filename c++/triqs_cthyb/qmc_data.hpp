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
#include <cmath>
#include "impurity_trace.hpp"
#include <triqs/gfs.hpp>
#include <triqs/mesh.hpp>
#include <triqs/det_manip.hpp>

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
      for (auto const &bl : range(delta.size())) {
#ifdef HYBRIDISATION_IS_COMPLEX
        auto delta_functor = delta_block_adaptor(delta[bl]);
#else
        if (!is_gf_real(delta[bl], 1e-10)) {
          if (p.verbosity >= 2) {
            std::cerr << "WARNING: The Delta(tau) block number " << bl << " is not real in tau space\n";
            std::cerr << "WARNING: max(Im[Delta(tau)]) = " << max_element(abs(imag(delta[bl].data()))) << "\n";
            std::cerr << "WARNING: Dissregarding the imaginary component in the calculation.\n";
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

    double compute_lang_firsov_ratio(std::vector<std::pair<time_pt, op_desc>> const& inserted, 
                                    std::vector<std::pair<time_pt, op_desc>> const& removed) const {
      if (!use_lang_firsov || K_n_size == 0) return 1.0;

      double d_w = 0.0;
      double beta = config.beta();

      struct perturb { time_pt t; int a; int S_op; int action; };
      std::vector<perturb> perts;
      for (auto const& p : inserted) perts.push_back({p.first, linindex.at({p.second.block_index, p.second.inner_index}), p.second.dagger ? 1 : -1, +1});
      for (auto const& p : removed)  perts.push_back({p.first, linindex.at({p.second.block_index, p.second.inner_index}), p.second.dagger ? 1 : -1, -1});

      // We need to sum over the background operators, excluding the ones to be removed
      // (because the removed ones are already in `config`, and we will account for them properly)
      
      //Compute the change in \sum_{a,b} k_n^{ab} \alpha_n^{ab}.
      // W = exp( 1/2 \sum_n k_n^{ab} \sum_{x \in a, y \in b} S_x S_y P_n(|t_x - t_y|...) )
      // versus w_{loc} = exp( \sum_n K_n \alpha_n ), Check factor of 2. 
      // \Delta W_exp = \sum_n \sum_a \sum_b K_n^{ab} \Delta \alpha_n^{ab}
      // \alpha^{ab}_{new} - \alpha^{ab}_{old}
      
      for (size_t i = 0; i < perts.size(); ++i) {
        auto p1 = perts[i];
        
        // 1. Cross terms with background operators (that are not being removed)
        for (auto const& [t_bg, op_bg] : config) {
          // Skip if this background operator is actually one of the ones being removed!
          bool is_removed = false;
          for (auto const& p_r : removed) {
             if (p_r.first == t_bg && p_r.second == op_bg) { is_removed = true; break; }
          }
          if (is_removed) continue;

          int b = linindex.at({op_bg.block_index, op_bg.inner_index});
          int S_bg = op_bg.dagger ? 1 : -1;
          
          double t_diff = double(p1.t - t_bg);
          if (t_diff < 0.0) t_diff += beta;
          double x = 2.0 * t_diff / beta - 1.0;
          
          for (int n = 0; n < K_n_size; ++n) {
             double P_n = std::legendre(n, x);
             // factor of 2 because k_n^{ab} term comes from both \alpha_n^{ab} and \alpha_n^{ba} if a!=b (assuming k_n is symmetric, which D0t is)
             // actually, the sum is over ALL \alpha, \beta without restriction.
             // So adding p1 creates two copies in the double sum: (p1, bg) and (bg, p1).
             double term = 2.0 * p1.action * p1.S_op * S_bg * P_n;
             d_w += K_n[p1.a][b][n] * term;
          }
        }
        
        // 2. Self term of the perturbation (p1 with itself)
        // Since action is +1 (insert) or -1 (remove), inserting adds p1*p1, removing subtracts p1*p1.
        double x_self = -1.0; // t_diff = 0 -> 2*0/beta - 1 = -1
        for (int n = 0; n < K_n_size; ++n) {
           double P_n = std::legendre(n, x_self);
           double term = p1.action * p1.S_op * p1.S_op * P_n; // action * (+1)
           d_w += K_n[p1.a][p1.a][n] * term;
        }
        
        // 3. Cross terms between perturbations
        for (size_t j = i + 1; j < perts.size(); ++j) {
           auto p2 = perts[j];
           double t_diff = double(p1.t - p2.t);
           if (t_diff < 0.0) t_diff += beta;
           double x = 2.0 * t_diff / beta - 1.0;
           // If we insert both: +1 * +1 = +1
           // If we remove both: we are removing their cross term: \alpha_old had (+1 * +1), \alpha_new has 0 -> diff = -1
           // If we insert p1 and remove p2, the \alpha_old had (bg, p2), \alpha_new has (bg, p1). The cross term (p1, p2) is never present!
           // Wait. If p1 is inserted, it only crosses with things in configuring AFTER p2 is removed. So it doesn't cross with p2!
           // So if p1.action != p2.action, their mutual cross-term in \Delta \alpha is ZERO.
           
           if (p1.action == p2.action) {
               for (int n = 0; n < K_n_size; ++n) {
                  double P_n = std::legendre(n, x);
                  double term = 2.0 * p1.action * p1.S_op * p2.S_op * P_n;
                  d_w += K_n[p1.a][p2.a][n] * term;
               }
           }
        }
      }

      return std::exp(d_w);
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
