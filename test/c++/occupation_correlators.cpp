// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
// SPDX-License-Identifier: GPL-3.0-or-later
// See LICENSE in the root of this distribution for details.

// The occupation correlators against exact diagonalization of the impurity and a discrete bath:
//   measure_O_tau and measure_nn_tau, which evaluate each configuration at points of tau, and
//   measure_D0_corr, which builds <O_i(tau) O_j(0)> from the occupation kinks (Q_tau, Q_conserved_tau).
//
// These are statistical checks with a fixed seed, not bit-reproducibility checks: the tolerances are
// several times the Monte Carlo noise, so they survive any change to how random numbers are drawn.
// Each test prints the largest deviation of every estimator, to compare their noise.

#include <triqs/test_tools/gfs.hpp>
#include <triqs/atom_diag/functions.hpp>
#include <triqs_cthyb/solver_core.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace triqs_cthyb;
using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::n;

namespace {

  // One bath site per (V, eps) for each spin-orbital of the impurity: Delta(iw) = sum_k V_k^2 / (iw - eps_k)
  struct bath_t {
    std::vector<double> V, eps;
  };

  // Delta(tau) = -sum_k V_k^2 e^{-eps_k tau} / (1 + e^{-beta eps_k}) on every diagonal element
  void set_delta(solver_core &solver, bath_t const &bath, double beta) {
    for (auto &delta : solver.Delta_tau()) {
      delta() = 0.0;
      for (auto const &tau : delta.mesh())
        for (long a = 0; a < delta.target_shape()[0]; ++a)
          for (size_t k = 0; k < bath.V.size(); ++k)
            delta[tau](a, a) -= bath.V[k] * bath.V[k] * std::exp(-bath.eps[k] * tau.value()) / (1.0 + std::exp(-beta * bath.eps[k]));
    }
  }

  // h_loc with the bath sites and their hybridization, diagonalized
  struct exact_t {
    triqs::atom_diag::atom_diag<false> ed;
    double beta;

    exact_t(many_body_op_t const &h_loc, gf_struct_t const &gf_struct, bath_t const &bath, double beta) : beta(beta) {
      fundamental_operator_set fops;
      many_body_op_t H = h_loc;
      for (auto const &[bl, size] : gf_struct)
        for (long a = 0; a < size; ++a) {
          fops.insert(bl, a);
          for (size_t k = 0; k < bath.V.size(); ++k) {
            auto const bath_bl = "bath_" + bl + "_" + std::to_string(k);
            fops.insert(bath_bl, a);
            H += bath.eps[k] * n(bath_bl, a) + bath.V[k] * (c_dag(bl, a) * c(bath_bl, a) + c_dag(bath_bl, a) * c(bl, a));
          }
        }
      ed = triqs::atom_diag::atom_diag<false>(H, fops);
    }

    // <A(tau) B(0)> = Tr[e^{-(beta - tau) H} A e^{-tau H} B] / Z
    double correlator(many_body_op_t const &A, many_body_op_t const &B, double tau) const {
      auto const A_mat = ed.get_op_mat(A), B_mat = ed.get_op_mat(B);
      double trace = 0.0, Z = 0.0;
      for (int b = 0; b < ed.n_subspaces(); ++b) {
        for (int m = 0; m < ed.get_subspace_dim(b); ++m) Z += std::exp(-beta * ed.get_eigenvalue(b, m));
        long const b1 = B_mat.connection(b);
        if (b1 < 0 || A_mat.connection(b1) != b) continue;
        for (int m = 0; m < ed.get_subspace_dim(b); ++m)
          for (int k = 0; k < ed.get_subspace_dim(b1); ++k)
            trace += std::exp(-(beta - tau) * ed.get_eigenvalue(b, m)) * A_mat.block_mat[b1](m, k) * std::exp(-tau * ed.get_eigenvalue(b1, k))
               * B_mat.block_mat[b](k, m);
      }
      return trace / Z;
    }
  };

  // max |value(tau) - exact(tau)| over every stride-th point of the mesh
  template <typename M, typename F, typename E> double max_deviation(M const &mesh, F const &value, E const &exact, long stride) {
    double dev = 0.0;
    for (auto const &tau : mesh)
      if (tau.index() % stride == 0) dev = std::max(dev, std::abs(value(tau) - exact(tau.value())));
    return dev;
  }

  // The kinks leave out the equal-time <O O'>, which the Python Solver adds from the density matrix
  double equal_time(solver_core const &solver, many_body_op_t const &op) {
    return std::real(triqs::atom_diag::trace_rho_op(solver.density_matrix(), op, solver.h_loc_diagonalization()));
  }

  constr_parameters_t constr_parameters(double beta, gf_struct_t const &gf_struct) {
    constr_parameters_t cp;
    cp.beta            = beta;
    cp.gf_struct       = gf_struct;
    cp.n_tau           = 10001; // fine enough for the nearest-point Delta(tau) not to bias the comparison
    cp.n_tau_bosonic   = 1001;
    cp.n_iw            = 100;
    cp.delta_interface = true;
    return cp;
  }

} // namespace

// One orbital, two bath sites per spin (the model of test/python/O_tau_ins.py). Every n_a commutes with h_loc, so
// measure_D0_corr gives the orbital-resolved Q_tau: the kink estimator against the sweep for the same correlators.
TEST(OccupationCorrelators, SingleOrbital) {
  double const beta = 2.1, U = 5.0, mu = 2.0;
  bath_t const bath{{2.0, 5.0}, {0.0, 4.0}};
  gf_struct_t const gf_struct{{"up", 1}, {"do", 1}};
  auto const n_up = n("up", 0), n_do = n("do", 0);
  auto const h_int = U * n_up * n_do, h_loc0 = -mu * (n_up + n_do);

  exact_t const exact(h_int + h_loc0, gf_struct, bath, beta);

  auto run = [&](auto configure) {
    solver_core solver(constr_parameters(beta, gf_struct));
    set_delta(solver, bath, beta);
    solve_parameters_t sp(h_int, 100000);
    sp.h_loc0          = h_loc0;
    sp.length_cycle    = 20;
    sp.n_warmup_cycles = 10000;
    sp.random_seed     = 3201;
    sp.measure_G_tau   = false;
    sp.verbosity       = 0;
    configure(sp);
    solver.solve(sp);
    return solver;
  };

  // The sweep: O_tau = <n_do(tau) n_up(0)> and every <n_a(tau) n_b(0)>
  auto sweep = run([&](solve_parameters_t &sp) {
    sp.measure_O_tau  = std::pair{n_up, n_do};
    sp.measure_nn_tau = true;
  });
  auto const &O  = *sweep.O_tau;
  auto const &nn = *sweep.nn_tau;
  double const dev_O_tau =
     max_deviation(O.mesh(), [&](auto const &tau) { return std::real(O[tau]); }, [&](double t) { return exact.correlator(n_do, n_up, t); }, 10);
  double const dev_nn_up_do = max_deviation(
     nn(0, 1).mesh(), [&](auto const &tau) { return std::real(nn(0, 1)[tau](0, 0)); }, [&](double t) { return exact.correlator(n_up, n_do, t); }, 10);
  double const dev_nn_up_up = max_deviation(
     nn(0, 0).mesh(), [&](auto const &tau) { return std::real(nn(0, 0)[tau](0, 0)); }, [&](double t) { return exact.correlator(n_up, n_up, t); }, 10);

  // The occupation kinks, completed with the equal-time part from the density matrix
  auto kinks = run([&](solve_parameters_t &sp) {
    sp.measure_D0_corr        = true;
    sp.measure_density_matrix = true;
    sp.use_norm_as_weight     = true;
  });
  ASSERT_TRUE(kinks.Q_tau.has_value());
  auto const &Q            = *kinks.Q_tau;
  double const dev_Q_up_do = max_deviation(
     Q(0, 1).mesh(), [&](auto const &tau) { return std::real(Q(0, 1)[tau](0, 0)) + equal_time(kinks, n_up * n_do); },
     [&](double t) { return exact.correlator(n_up, n_do, t); }, 10);
  double const dev_Q_up_up = max_deviation(
     Q(0, 0).mesh(), [&](auto const &tau) { return std::real(Q(0, 0)[tau](0, 0)) + equal_time(kinks, n_up * n_up); },
     [&](double t) { return exact.correlator(n_up, n_up, t); }, 10);

  std::cout << "max |QMC - ED|:  O_tau " << dev_O_tau << ",  nn_tau[up,do] " << dev_nn_up_do << ",  nn_tau[up,up] " << dev_nn_up_up
            << ",  D0_corr Q_tau[up,do] " << dev_Q_up_do << ",  Q_tau[up,up] " << dev_Q_up_up << std::endl;

  EXPECT_LT(dev_O_tau, 0.02);
  EXPECT_LT(dev_nn_up_do, 0.02);
  EXPECT_LT(dev_nn_up_up, 0.02);
  EXPECT_LT(dev_Q_up_do, 0.02);
  EXPECT_LT(dev_Q_up_up, 0.02);
}

namespace {
  // Two orbitals with the full Kanamori interaction, spin flip and pair hopping included
  many_body_op_t kanamori(double U, double J) {
    many_body_op_t h_int;
    for (int a = 0; a < 2; ++a) h_int += U * n("up", a) * n("do", a);
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) {
        if (a == b) continue;
        h_int += (U - 2 * J) * n("up", a) * n("do", b);
        if (a < b) h_int += (U - 3 * J) * (n("up", a) * n("up", b) + n("do", a) * n("do", b));
        h_int += -J * c_dag("up", a) * c("do", a) * c_dag("do", b) * c("up", b);
        h_int += J * c_dag("up", a) * c_dag("do", a) * c("do", b) * c("up", b);
      }
    return h_int;
  }

  many_body_op_t total_density(int n_orb) {
    many_body_op_t N;
    for (auto const &bl : {"up", "do"})
      for (int a = 0; a < n_orb; ++a) N += n(bl, a);
    return N;
  }
} // namespace

// Operators that change occupations, so that the trace with them runs through other blocks of h_loc than the bare one:
// the spin flips S^+ = sum_a c^dagger_{up,a} c_{do,a} and S^- = (S^+)^dagger, and, for two orbitals, the orbital
// exchange T = sum_s c^dagger_{s,1} c_{s,0}. measure_O_tau = (O_1, O_2) gives <O_2(tau) O_1(0)>.
TEST(OccupationCorrelators, OffDiagonalOperators) {
  auto spin_plus = [](int n_orb) {
    many_body_op_t S;
    for (int a = 0; a < n_orb; ++a) S += c_dag("up", a) * c("do", a);
    return S;
  };
  auto measure = [](double beta, gf_struct_t const &gf_struct, bath_t const &bath, many_body_op_t const &h_int, many_body_op_t const &h_loc0,
                    many_body_op_t const &O1, many_body_op_t const &O2, long n_cycles, long seed) {
    solver_core solver(constr_parameters(beta, gf_struct));
    set_delta(solver, bath, beta);
    solve_parameters_t sp(h_int, n_cycles);
    sp.h_loc0          = h_loc0;
    sp.length_cycle    = 50;
    sp.n_warmup_cycles = 5000;
    sp.random_seed     = seed;
    sp.measure_G_tau   = false;
    sp.verbosity       = 0;
    sp.measure_O_tau   = std::pair{O1, O2};
    solver.solve(sp);
    return *solver.O_tau;
  };

  // One orbital: <S^+(tau) S^-(0)>, a single term that takes each block to another
  {
    double const beta = 2.1, U = 5.0, mu = 2.0;
    bath_t const bath{{2.0, 5.0}, {0.0, 4.0}};
    gf_struct_t const gf_struct{{"up", 1}, {"do", 1}};
    many_body_op_t const h_int = U * n("up", 0) * n("do", 0), h_loc0 = -mu * total_density(1);
    exact_t const exact(h_int + h_loc0, gf_struct, bath, beta);
    auto const S_plus = spin_plus(1), S_minus = dagger(S_plus);

    auto O        = measure(beta, gf_struct, bath, h_int, h_loc0, S_minus, S_plus, 50000, 4127);
    double const dev = max_deviation(O.mesh(), [&](auto const &tau) { return std::real(O[tau]); },
                                     [&](double t) { return exact.correlator(S_plus, S_minus, t); }, 10);
    std::cout << "one orbital, max |QMC - ED| of <S+(tau) S-(0)>: " << dev << std::endl;
    EXPECT_LT(dev, 0.02);
  }

  // Two Kanamori orbitals: S^+ and T, sums of terms that take a block to different blocks
  {
    double const beta = 2.0, U = 2.0, J = 0.3, mu = (3 * U - 5 * J) / 2;
    bath_t const bath{{0.8}, {0.3}};
    gf_struct_t const gf_struct{{"up", 2}, {"do", 2}};
    auto const h_int = kanamori(U, J), h_loc0 = -mu * total_density(2);
    exact_t const exact(h_int + h_loc0, gf_struct, bath, beta);

    auto const S_plus = spin_plus(2), S_minus = dagger(S_plus);
    auto const O_S    = measure(beta, gf_struct, bath, h_int, h_loc0, S_minus, S_plus, 30000, 6011);
    double const dev_S = max_deviation(O_S.mesh(), [&](auto const &tau) { return std::real(O_S[tau]); },
                                       [&](double t) { return exact.correlator(S_plus, S_minus, t); }, 20);

    auto const T   = c_dag("up", 1) * c("up", 0) + c_dag("do", 1) * c("do", 0);
    auto const O_T = measure(beta, gf_struct, bath, h_int, h_loc0, dagger(T), T, 30000, 6029);
    double const dev_T = max_deviation(O_T.mesh(), [&](auto const &tau) { return std::real(O_T[tau]); },
                                       [&](double t) { return exact.correlator(T, dagger(T), t); }, 20);

    std::cout << "two orbitals, max |QMC - ED| of <S+(tau) S-(0)>: " << dev_S << ",  <T(tau) T+(0)>: " << dev_T << std::endl;
    EXPECT_LT(dev_S, 0.03);
    EXPECT_LT(dev_T, 0.03);
  }
}

// Two orbitals with the full Kanamori interaction: spin flip and pair hopping mix the occupations, so no single n_a
// commutes with h_loc and nn_tau comes from the sweep with n_a off-diagonal in the eigenbasis. measure_D0_corr gives
// the correlators of the conserved combinations (N_up and N_do, in some basis) in Q_conserved_tau.
TEST(OccupationCorrelators, TwoOrbitalKanamori) {
  double const beta = 2.0, U = 2.0, J = 0.3, mu = (3 * U - 5 * J) / 2;
  bath_t const bath{{0.8}, {0.3}};
  gf_struct_t const gf_struct{{"up", 2}, {"do", 2}};

  auto const h_int = kanamori(U, J), N = total_density(2), h_loc0 = -mu * N;

  exact_t const exact(h_int + h_loc0, gf_struct, bath, beta);

  solver_core solver(constr_parameters(beta, gf_struct));
  set_delta(solver, bath, beta);
  solve_parameters_t sp(h_int, 50000);
  sp.h_loc0                 = h_loc0;
  sp.length_cycle           = 50;
  sp.n_warmup_cycles        = 5000;
  sp.random_seed            = 5279;
  sp.measure_G_tau          = false;
  sp.verbosity              = 0;
  sp.measure_O_tau          = std::pair{N, N};
  sp.measure_nn_tau         = true;
  sp.measure_D0_corr        = true;
  sp.measure_density_matrix = true;
  sp.use_norm_as_weight     = true;
  solver.solve(sp);

  // The sweep, for every pair of spin-orbitals
  double dev_nn  = 0.0;
  auto const &nn = *solver.nn_tau;
  std::vector<std::string> const blocks{"up", "do"};
  for (int b1 = 0; b1 < 2; ++b1)
    for (int b2 = 0; b2 < 2; ++b2)
      for (int a1 = 0; a1 < 2; ++a1)
        for (int a2 = 0; a2 < 2; ++a2)
          dev_nn = std::max(dev_nn, max_deviation(
                                       nn(b1, b2).mesh(), [&](auto const &tau) { return std::real(nn(b1, b2)[tau](a1, a2)); },
                                       [&](double t) { return exact.correlator(n(blocks[b1], a1), n(blocks[b2], a2), t); }, 20));
  auto const &O = *solver.O_tau;
  double const dev_O_tau =
     max_deviation(O.mesh(), [&](auto const &tau) { return std::real(O[tau]); }, [&](double t) { return exact.correlator(N, N, t); }, 20);

  // The occupation kinks of the conserved combinations
  auto const &ops = solver.conserved_density_operators;
  ASSERT_EQ(ops.size(), 2);
  auto const &Q = *solver.Q_conserved_tau;
  double dev_Q  = 0.0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      dev_Q = std::max(dev_Q, max_deviation(
                                 Q.mesh(), [&](auto const &tau) { return std::real(Q[tau](i, j)) + equal_time(solver, ops[i] * ops[j]); },
                                 [&](double t) { return exact.correlator(ops[i], ops[j], t); }, 20));

  std::cout << "max |QMC - ED|:  nn_tau " << dev_nn << ",  O_tau = <N N> " << dev_O_tau << ",  D0_corr Q_conserved_tau " << dev_Q
            << std::endl;

  EXPECT_LT(dev_nn, 0.02);
  EXPECT_LT(dev_O_tau, 0.05);
  EXPECT_LT(dev_Q, 0.05);
}

MAKE_MAIN;
