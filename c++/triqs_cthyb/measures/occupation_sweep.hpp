#pragma once

#include <vector>
#include <nda/nda.hpp>
#include <mpi/mpi.hpp>

#include "../qmc_data.hpp"

namespace triqs_cthyb {

  /// <A_i(tau) B_j(0)> for operators diagonal in the occupation basis (n_a, N, S_z, n_a n_b, ...), exactly at every point
  /// of a bosonic mesh of n_tau points, configuration by configuration.
  ///
  /// B_j sits at the trace boundary tau = 0 (translation invariance) and A_i sweeps the trace. Between the trace operators
  /// at t_k and t_{k+1}, in the block of h_diag the trace runs through there,
  ///   <A_i(tau) B_j(0)> = sum_mn c_mn e^{-(t_{k+1} - tau) E_m} e^{-(tau - t_k) E_n},   c_mn = (A_i)_mn (R_k B_j L_k)_nm,
  /// with R_k the trace from 0 to t_k and L_k the trace from t_{k+1} to beta, divided by the trace. Each term is a
  /// constant (E_m = E_n) or an exponential that decays away from one end of the interval at the rate |E_m - E_n|. It is
  /// not evaluated on the mesh but deposited at the two ends of its interval, in an accumulator per rate and direction;
  /// collect() spreads the deposits over the mesh with one recursion per accumulator. A measurement then costs nothing
  /// per mesh point.
  class occupation_sweep {
    public:
    occupation_sweep(qmc_data const &data, long n_tau, std::vector<many_body_op_t> const &ops_tau,
                     std::vector<many_body_op_t> const &ops_0);

    void accumulate(mc_weight_t s);

    /// (tau index, i, j) -> <A_i(tau) B_j(0)>, all-reduced over c
    nda::array<mc_weight_t, 3> collect(mpi::communicator const &c);

    private:
    // One term (m, n) of a block. slot 0 is the constants; slot -1 an exponential evaluated directly on the mesh, once
    // the accumulators would exceed MAX_ACCUMULATED elements.
    struct term_t {
      int m, n;
      int slot;
      double rate;   // |E_m - E_n|, 0 for a constant
      bool backward; // E_m > E_n: decays from t_{k+1} towards t_k
    };
    struct slot_t {
      double decay; // e^{-rate dtau}
      bool backward;
    };
    static constexpr long MAX_ACCUMULATED = 1L << 22;

    void deposit(term_t const &term, long p0, long p1, double t_left, double t_right, double E_m, double E_n);

    qmc_data const &data;
    long n_tau;
    double dtau;
    long n_A, n_B;
    std::vector<atom_diag::op_block_mat_t> A_mat, B_mat; // in the eigenbasis, by block of h_diag
    std::vector<std::vector<term_t>> terms;              // by block of h_diag
    std::vector<slot_t> slots;
    nda::array<mc_weight_t, 3> accumulated; // (slot, tau index, i * n_B + j)
    nda::array<mc_weight_t, 2> direct;      // (tau index, i * n_B + j), empty without direct terms
    mc_weight_t average_sign = 0;

    // Work space, kept between measurements
    std::vector<int> block;
    std::vector<double> t;
    std::vector<long> first_point;
    std::vector<std::vector<double>> decays;
    std::vector<matrix_t> R, L, Y;
    matrix_t evolved, RB;
    std::vector<mc_weight_t> coefficients;
  };

} // namespace triqs_cthyb
