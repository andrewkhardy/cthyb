/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2017, H. U.R. Strand, M. Ferrero and O. Parcollet
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

#include <climits>

#include "./config.hpp"
#include "./types.hpp"
#include "./configuration.hpp"

namespace triqs_cthyb {

  using namespace triqs::operators;
  using indices_map_t = std::map<triqs::operators::indices_t, triqs::operators::indices_t>;

  /// Parameters used for constructing the solver class.
  struct constr_parameters_t {

    /// Inverse temperature \f$ \beta \f$.
    double beta;

    /// Structure of the Green's function (names and sizes of blocks).
    gf_struct_t gf_struct;

    /// Number of Matsubara frequencies.
    int n_iw = 1025;

    /// Number of imaginary-time points.
    int n_tau = 10001;

    /// Number of imaginary-time points of the dynamical interactions and their correlators.
    int n_tau_bosonic = 10001;

    /// Number of Legendre polynomials.
    int n_l = 50;

    /// Use \f$ \Delta(\tau) \f$ and \f$ h_{loc0} \f$ as input instead of \f$ G_0(i\omega) \f$.
    bool delta_interface = false;

    bool operator==(constr_parameters_t const &) const = default;

    /// Write constr_parameters_t to hdf5.
    friend void h5_write(h5::group h5group, std::string subgroup_name, constr_parameters_t const &sp);

    /// Read constr_parameters_t from hdf5.
    friend void h5_read(h5::group h5group, std::string subgroup_name, constr_parameters_t &sp);
  };

  /// Parameters passed to the solve method of the solver class.
  struct solve_parameters_t {

    /// Interacting part of the atomic Hamiltonian.
    many_body_op_t h_int;

    /// Number of QMC cycles.
    long n_cycles;

    /// Partition method.
    std::string partition_method = "autopartition";

    /// Quantum numbers.
    std::vector<many_body_op_t> quantum_numbers = {};

    /// Restrict local Hilbert space to states with at least this number of particles.
    int loc_n_min = 0;

    /// Restrict local Hilbert space to states with at most this number of particles.
    int loc_n_max = INT_MAX;

    /// Length of a single QMC cycle.
    long length_cycle = 50;

    /// Number of cycles for thermalization.
    long n_warmup_cycles = 5000;

    /// Seed for random number generator.
    long random_seed = 34788 + 928374 * mpi::communicator().rank();

    /// Name of random number generator.
    std::string random_name = "";

    /// Maximum runtime in seconds, use -1 to set infinite.
    long max_time = -1;

    /// Verbosity level.
    int verbosity = ((mpi::communicator().rank() == 0) ? 3 : 0); // silence the slave nodes

    /// Add shifting an operator as a move?
    bool move_shift = true;

    /// Add double insertions as a move? Unset: off when the pair moves alone are ergodic (every block of size 1, every density
    /// commuting with h_loc and density-density dynamical vertices), on otherwise.
    std::optional<bool> move_double = {};

    /// Probability that the pair insertions and removals use the Pauli proposal, which puts the c between its c^dagger and the
    /// nearest operator of the same inner index, rather than the uniform one. Unset: 1 under the conditions where move_double
    /// is off by default (the segment insertions and removals of CT-SEG), 0 otherwise.
    std::optional<double> pauli_prob = {};

    /// Calculate the full trace or use an estimate?
    bool use_trace_estimator = false;

    /// Resum the density-density dynamical vertices that commute with h_loc analytically (Lang-Firsov)?
    /// All other vertices, or all of them if false, are sampled stochastically.
    bool lang_firsov = true;

    /// Number of Legendre coefficients of the Lang-Firsov kernel and of the dynamical correlators.
    int dyn_n_l = 50;

    /// Measure \f$ G(\tau) \f$? Hermiticity \f$ G_{ij}(\tau) = G_{ji}^*(\tau) \f$ is enforced.
    bool measure_G_tau = true;

    /// Measure \f$ G_l \f$ (Legendre)? No hermiticity is enforced.
    bool measure_G_l = false;

    /// Measure \f$ \langle O_2(\tau) O_1(0) \rangle \f$ for measure_O_tau = (O_1, O_2), bosonic operators (an even number
    /// of c and c^dagger in every term), whether or not they commute with h_loc. Measured at the nodes of a DLR grid
    /// (dlr_w_max, dlr_eps), with the result in O_dlr and, on n_tau points, in O_tau. Operators that change the block of
    /// h_loc (S^+, c^dagger_a c_b, ...) cost more.
    std::optional<std::pair<many_body_op_t, many_body_op_t>> measure_O_tau = {};

    /// Measure \f$ \langle n_a(\tau) n_b(0) \rangle \f$ for every pair of spin-orbitals, whether or not the n_a commute
    /// with h_loc. If they do and measure_D0_corr and measure_density_matrix are set, nn_tau is Q_tau from the occupation
    /// kinks; otherwise it is measured like measure_O_tau, with the result in nn_dlr and, on n_tau_bosonic points, in nn_tau.
    bool measure_nn_tau = false;

    /// Highest frequency of the DLR grid of measure_O_tau and measure_nn_tau. The correlators must have no spectral weight
    /// above it; check by doubling it.
    double dlr_w_max = 20.0;

    /// Accuracy of the DLR grid of measure_O_tau and measure_nn_tau.
    double dlr_eps = 1e-10;

    /// Measure density-density correlators from the occupation kinks: Q_conserved_tau for the density combinations that
    /// commute with h_loc, and Q_tau if every n_a does. The equal-time part is added by the Python Solver with the density matrix.
    bool measure_D0_corr = false;

    /// Measure \f$ G^{(2)}(\tau,\tau',\tau'') \f$ with three fermionic times.
    bool measure_G2_tau = false;

    /// Measure \f$ G^{(2)}(i\nu,i\nu',i\nu'') \f$ with three fermionic frequencies.
    bool measure_G2_iw = false;

    /// Measure \f$ G^{(2)}(i\nu,i\nu',i\nu'') \f$ with three fermionic frequencies.
    bool measure_G2_iw_nfft = false;

    /// Measure \f$ G^{(2)}(i\omega,i\nu,i\nu') \f$ in the particle-particle channel.
    bool measure_G2_iw_pp = false;

    /// Measure \f$ G^{(2)}(i\omega,i\nu,i\nu') \f$ in the particle-particle channel.
    bool measure_G2_iw_pp_nfft = false;

    /// Measure \f$ G^{(2)}(i\omega,i\nu,i\nu') \f$ in the particle-hole channel.
    bool measure_G2_iw_ph = false;

    /// Measure \f$ G^{(2)}(i\omega,i\nu,i\nu') \f$ in the particle-hole channel.
    bool measure_G2_iw_ph_nfft = false;

    /// Measure \f$ G^{(2)}(i\omega,l,l') \f$ in the particle-particle channel.
    bool measure_G2_iwll_pp = false;

    /// Measure \f$ G^{(2)}(i\omega,l,l') \f$ in the particle-hole channel.
    bool measure_G2_iwll_ph = false;

    /// Order of block indices in the definition of \f$ G^{(2)} \f$.
    block_order measure_G2_block_order = block_order::AABB;

    /// List of block index pairs of \f$ G^{(2)} \f$ to measure.
    std::set<std::pair<std::string, std::string>> measure_G2_blocks = {};

    /// Number of imaginary-time slices for the \f$ G^{(2)} \f$ measurement.
    int measure_G2_n_tau = 10;

    /// Number of bosonic Matsubara frequencies for the \f$ G^{(2)} \f$ measurement.
    int measure_G2_n_bosonic = 30;

    /// Number of fermionic Matsubara frequencies for the \f$ G^{(2)} \f$ measurement.
    int measure_G2_n_fermionic = 30;

    /// Number of Legendre coefficients for the \f$ G^{(2)}(i\omega,l,l') \f$ measurement.
    int measure_G2_n_l = 20;

    /// NFFT buffer size for the \f$ G^{(2)}(i\omega,l,l') \f$ measurement.
    int measure_G2_iwll_nfft_buf_size = 100;

    /// NFFT buffer sizes for different blocks.
    std::map<std::string, long> nfft_buf_sizes = {};

    /// Measure perturbation order?
    bool measure_pert_order = false;

    /// Measure the reduced impurity density matrix?
    bool measure_density_matrix = false;

    /// Use the norm of the density matrix in the weight (instead of the trace)?
    bool use_norm_as_weight = false;

    /// Initial configuration of the run (advanced, use with care).
    std::optional<configuration> initial_configuration = {};

    /// Analyse performance of the trace computation with histograms (developers only)?
    bool performance_analysis = false;

    /// Operator insertion/removal probabilities for different blocks.
    std::map<std::string, double> proposal_prob = {};

    /// List of global moves (with their names). Each move is specified with an index substitution dictionary.
    std::map<std::string, indices_map_t> move_global = {};

    /// Overall probability of the global moves.
    double move_global_prob = 0.05;

    /// Substitute every operator mapped by a global move, instead of a random subset?
    bool move_global_full = false;

    /// Threshold below which imaginary components of \f$ \Delta \f$ and \f$ h_{loc} \f$ are set to zero.
    double imag_threshold = 1.e-13;

    /// The maximum size of the determinant matrix before a resize.
    int det_init_size = 100;

    /// Maximum number of operations before testing the accuracy of \f$ \det(M) \f$ and \f$ M^{-1} \f$.
    int det_n_operations_before_check = 100;

    /// Threshold for determinant precision warnings.
    double det_precision_warning = 1.e-8;

    /// Threshold for determinant precision error.
    double det_precision_error = 1.e-5;

    /// Bound for the determinant matrix being singular (if \f$ < 0 \f$, checks for subnormal numbers).
    double det_singular_threshold = -1;

    bool operator==(solve_parameters_t const &) const = default;

    /// Write solve_parameters_t to hdf5
    friend void h5_write(h5::group h5group, std::string subgroup_name, solve_parameters_t const &sp);

    /// Read solve_parameters_t from hdf5
    friend void h5_read(h5::group h5group, std::string subgroup_name, solve_parameters_t &sp);

    /// Threshold below which off-diagonal components of \f$ h_{loc} \f$ are set to zero.
    double off_diag_threshold = 0.0;

    /// Quadratic part of the local Hamiltonian. Must be provided if the \f$ \Delta \f$ interface is used.
    std::optional<many_body_op_t> h_loc0 = {};
  };

  /// A struct combining both constr_params_t and solve_params_t
  struct params_t : constr_parameters_t, solve_parameters_t {
    params_t(constr_parameters_t constr_parameters_, solve_parameters_t solve_parameters_)
       : constr_parameters_t(constr_parameters_), solve_parameters_t(solve_parameters_) {}
  };
} // namespace triqs_cthyb
