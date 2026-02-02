// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
// SPDX-License-Identifier: GPL-3.0-or-later
// See LICENSE in the root of this distribution for details.

#include <cmath>
#include <triqs/test_tools/gfs.hpp>
#include <triqs_cthyb/solver_core.hpp>

using triqs::operators::n;
using namespace triqs_cthyb;

TEST(CTHYB, Spin_Spin) {

  mpi::communicator c; // Start the mpi

  double beta         = 10.0;
  double U            = 4.0;
  double mu           = 2.0;
  double epsilon      = 0.3;
  int n_cycles        = 10000;
  int n_warmup_cycles = 1000;
  int length_cycle    = 50;
  int random_seed     = 23488;
  int n_iw            = 1025;
  int n_tau           = 10001;
  int n_tau_bosonic   = 10001;

  // Prepare the construction parameters
  constr_parameters_t param_constructor;
  param_constructor.beta            = beta;
  param_constructor.gf_struct       = {{"up", 1}, {"down", 1}};
  param_constructor.n_iw            = n_iw;
  param_constructor.n_tau           = n_tau;
  param_constructor.n_tau_bosonic   = n_tau_bosonic;
  param_constructor.delta_interface = true; // Use Delta_tau interface

  // Create solver instance
  solver_core Solver(param_constructor);

  // Prepare Delta(tau)
  nda::clef::placeholder<0> om_;
  auto Delta_w   = gf<imfreq>({beta, Fermion, n_iw}, {1, 1});
  auto Delta_tau = gf<imtime>({beta, Fermion, n_tau}, {1, 1});
  Delta_w(om_) << 1.0 / (om_ - epsilon) + 1.0 / (om_ + epsilon);
  Delta_tau()           = fourier(Delta_w);
  Solver.Delta_tau()[0] = Delta_tau;
  Solver.Delta_tau()[1] = Delta_tau;

  // Prepare spin-spin interaction
  double l  = 1.0; // electron boson coupling
  double w0 = 1.0; // screening frequency
  auto J0w  = gf<imfreq>({beta, Boson, n_iw}, {1, 1});
  auto D0w  = gf<imfreq>({beta, Boson, n_iw}, {1, 1});
  auto D0t  = gf<imtime>({beta, Boson, n_tau_bosonic}, {1, 1});
  auto J0t  = gf<imtime>({beta, Boson, n_tau_bosonic}, {1, 1});
  J0w(om_) << 4 * l * l * w0 / (om_ * om_ - w0 * w0);
  D0w(om_) << l * l * w0 / (om_ * om_ - w0 * w0);
  D0t()                 = fourier(D0w);
  J0t()                 = fourier(J0w);
  Solver.D0_tau()(0, 0) = D0t;
  Solver.D0_tau()(0, 1) = -D0t;
  Solver.D0_tau()(1, 0) = -D0t;
  Solver.D0_tau()(1, 1) = D0t;
  Solver.Jperp_tau()    = J0t;

  // Solve parameters
  solve_parameters_t param_solve(U * n("up", 0) * n("down", 0), n_cycles);
  param_solve.h_loc0            = -mu * (n("up", 0) + n("down", 0));
  param_solve.n_warmup_cycles   = n_warmup_cycles;
  param_solve.length_cycle      = length_cycle;
  param_solve.random_seed       = random_seed;
  param_solve.measure_G_tau     = true;
  // Dynamical interaction moves are automatically enabled when D0_tau or Jperp_tau are non-zero

  // Solve
  std::cout << "Solving with dynamical interactions D0_tau and Jperp_tau..." << std::endl;
  Solver.solve(param_solve);

  // Save the results
  if (c.rank() == 0) {
    h5::file out_file("spin_spin.out.h5", 'w');
    h5_write(out_file, "G_tau", *Solver.G_tau);
  }

  // Basic sanity check: Green's function should be negative at tau=0+ and tau=beta-
  auto &G_up = (*Solver.G_tau)[0];
  auto &G_down = (*Solver.G_tau)[1];
  
  // G(tau=0+) should be approximately -<n> (negative)
  std::cout << "G_up(0) = " << G_up.data()(0, 0, 0) << std::endl;
  std::cout << "G_down(0) = " << G_down.data()(0, 0, 0) << std::endl;
  std::cout << "G_up(beta-) = " << G_up.data()(n_tau - 1, 0, 0) << std::endl;
  std::cout << "G_down(beta-) = " << G_down.data()(n_tau - 1, 0, 0) << std::endl;
  
  // Sanity checks
  EXPECT_LT(real(G_up.data()(0, 0, 0)), 0);     // G(tau=0+) < 0
  EXPECT_LT(real(G_down.data()(0, 0, 0)), 0);   // G(tau=0+) < 0
  EXPECT_GT(real(G_up.data()(n_tau - 1, 0, 0)), -1);  // G(tau=beta-) > -1
  EXPECT_GT(real(G_down.data()(n_tau - 1, 0, 0)), -1);
  
  std::cout << "Spin-spin dynamical interaction test passed!" << std::endl;
}
MAKE_MAIN;