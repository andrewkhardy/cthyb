// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
// SPDX-License-Identifier: GPL-3.0-or-later
// See LICENSE in the root of this distribution for details.

#include <cmath>
#include <triqs/test_tools/gfs.hpp>
#include <triqs_cthyb/solver_core.hpp>

using triqs::operators::n;
using namespace triqs_cthyb;

// Single-orbital test with dynamical spin-spin interactions (D0_tau + Jperp_tau).
// Uses a simple analytical bath: Delta(iw) = 1/(iw - eps) + 1/(iw + eps)
// and a bosonic propagator: J(iw) = 4*l^2*w0/(iw^2 - w0^2)
// This tests that the dynamical interaction machinery (Jperp spin-flip +
// D0 density-density retarded interactions) produces correct results.
//
// To regenerate the reference file spin_spin.ref.h5:
//   1. Build and run this test once (produces spin_spin.out.h5)
//   2. Verify the results are physically reasonable
//   3. Copy spin_spin.out.h5 -> spin_spin.ref.h5

TEST(CTHYB, Spin_Spin) {

  mpi::communicator c;
  int rank = c.rank();

  // Physical parameters
  double beta    = 10.0;
  double U       = 4.0;
  double mu      = U / 2.0; // half-filling
  double epsilon = 0.3;     // bath level
  double l       = 1.0;     // electron-boson coupling
  double w0      = 1.0;     // screening frequency

  // Solver parameters — fixed seed for reproducibility
  int n_cycles        = 10000;
  int n_warmup_cycles = 1000;
  int length_cycle    = 50;
  int random_seed     = 23488;
  int n_iw            = 1025;
  int n_tau           = 10001;
  int n_tau_bosonic   = 10001;

  // Construct solver with Delta_tau interface
  constr_parameters_t cparams;
  cparams.beta            = beta;
  cparams.gf_struct       = {{"up", 1}, {"down", 1}};
  cparams.n_iw            = n_iw;
  cparams.n_tau           = n_tau;
  cparams.n_tau_bosonic   = n_tau_bosonic;
  cparams.delta_interface = true;

  solver_core solver(cparams);

  // Prepare Delta(tau): symmetric two-pole bath
  nda::clef::placeholder<0> om_;
  auto Delta_w = gf<imfreq>({beta, Fermion, n_iw}, {1, 1});
  Delta_w(om_) << 1.0 / (om_ - epsilon) + 1.0 / (om_ + epsilon);
  auto Delta_tau = gf<imtime>({beta, Fermion, n_tau}, {1, 1});
  Delta_tau() = fourier(Delta_w);

  solver.Delta_tau()[0] = Delta_tau; // up
  solver.Delta_tau()[1] = Delta_tau; // down

  // Prepare dynamical interactions from bosonic propagator
  auto J0w = gf<imfreq>({beta, Boson, n_iw}, {1, 1});
  auto D0w = gf<imfreq>({beta, Boson, n_iw}, {1, 1});
  J0w(om_) << 4 * l * l * w0 / (om_ * om_ - w0 * w0);
  D0w(om_) << l * l * w0 / (om_ * om_ - w0 * w0);

  auto J0t = gf<imtime>({beta, Boson, n_tau_bosonic}, {1, 1});
  auto D0t = gf<imtime>({beta, Boson, n_tau_bosonic}, {1, 1});
  J0t() = fourier(J0w);
  D0t() = fourier(D0w);

  // Jperp: spin-flip interaction. Stored in the diagonal block pair.
  // gf_struct = {up, down} so block 0 = "up".
  solver.Jperp_tau()(0, 0) = J0t;

  // D0: density-density retarded interaction
  // Sz*Sz decomposition: same-spin = +D0, opposite-spin = -D0
  solver.D0_tau()(0, 0) = D0t;   // up-up
  solver.D0_tau()(0, 1) = -D0t;  // up-down
  solver.D0_tau()(1, 0) = -D0t;  // down-up
  solver.D0_tau()(1, 1) = D0t;   // down-down

  // Solve
  auto H = U * n("up", 0) * n("down", 0);
  solve_parameters_t sparams(H, n_cycles);
  sparams.h_loc0            = -mu * (n("up", 0) + n("down", 0));
  sparams.n_warmup_cycles   = n_warmup_cycles;
  sparams.length_cycle      = length_cycle;
  sparams.random_seed       = random_seed;
  sparams.random_name       = "";
  sparams.measure_G_tau     = true;
  sparams.measure_pert_order = true;

  std::cout << "Solving with dynamical spin-spin interactions (D0_tau + Jperp_tau)..." << std::endl;
  solver.solve(sparams);

  auto &G_tau = *solver.G_tau;

  // Save output
  if (rank == 0) {
    h5::file out_file("spin_spin.out.h5", 'w');
    h5_write(out_file, "G_up", G_tau[0]);
    h5_write(out_file, "G_down", G_tau[1]);
  }

  // Compare against reference
  gf<imtime> g;
  if (rank == 0) {
    h5::file ref_file("spin_spin.ref.h5", 'r');
    h5_read(ref_file, "G_up", g);
    EXPECT_GF_NEAR(g, G_tau[0]);
    h5_read(ref_file, "G_down", g);
    EXPECT_GF_NEAR(g, G_tau[1]);
  }

  std::cout << "Spin-spin dynamical interaction test passed!" << std::endl;
}
MAKE_MAIN;