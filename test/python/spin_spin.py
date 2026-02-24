# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.

# Single-orbital test with dynamical spin-spin interactions (D0_tau + Jperp_tau).
# Uses a simple analytical bath: Delta(iw) = 1/(iw - eps) + 1/(iw + eps)
# and a bosonic propagator: J(iw) = 4*l^2*w0/(iw^2 - w0^2)
#
# This mirrors the C++ spin_spin test but exercises the Python interface,
# especially the block2_gf Jperp_tau and D0_tau assignment.
#
# To regenerate spin_spin.ref.h5:
#   1. Run this script once (produces spin_spin.out.h5)
#   2. Verify the results are physically reasonable
#   3. Copy spin_spin.out.h5 -> spin_spin.ref.h5

import triqs.utility.mpi as mpi
from triqs.gf import *
from triqs.operators import n
from h5 import HDFArchive
from triqs.utility.comparison_tests import *
from triqs_cthyb import SolverCore as Solver

# Physical parameters (same as C++ test)
beta    = 10.0
U       = 4.0
mu      = U / 2.0   # half-filling
epsilon = 0.3        # bath level
l       = 1.0        # electron-boson coupling
w0      = 1.0        # screening frequency

# Discretization
n_iw          = 1025
n_tau         = 10001
n_tau_bosonic = 10001

# gf_struct — note 'down' before 'up' (tests that Jperp finds the
# non-zero diagonal block regardless of block ordering)
gf_struct = [('down', 1), ('up', 1)]

# Construct solver
S = Solver(beta=beta, gf_struct=gf_struct, n_iw=n_iw, n_tau=n_tau,
           n_tau_bosonic=n_tau_bosonic, delta_interface=True)

# Hybridization: symmetric two-pole bath
Delta_iw = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
Delta_iw << 1.0 / (iOmega_n - epsilon) + 1.0 / (iOmega_n + epsilon)
S.Delta_tau << Fourier(Delta_iw)

# Bosonic propagators
J0_iw = GfImFreq(indices=[0], beta=beta, n_points=n_iw, statistic='Boson')
D0_iw = GfImFreq(indices=[0], beta=beta, n_points=n_iw, statistic='Boson')
J0_iw << 4 * l**2 * w0 / (iOmega_n**2 - w0**2)
D0_iw << l**2 * w0 / (iOmega_n**2 - w0**2)
J0_tau = GfImTime(indices=[0], beta=beta, n_points=n_tau_bosonic, statistic='Boson')
D0_tau = GfImTime(indices=[0], beta=beta, n_points=n_tau_bosonic, statistic='Boson')
J0_tau << Fourier(J0_iw)
D0_tau << Fourier(D0_iw)

# Jperp: spin-flip interaction (stored in the "up"-"up" diagonal block pair)
S.Jperp_tau["up", "up"] << J0_tau

# D0: density-density retarded interaction
# Sz*Sz decomposition: same-spin = +D0, opposite-spin = -D0
S.D0_tau["up", "up"]     << D0_tau
S.D0_tau["down", "down"] << D0_tau
S.D0_tau["up", "down"]   << -1.0 * D0_tau
S.D0_tau["down", "up"]   << -1.0 * D0_tau

# Solve parameters — fixed seed for reproducibility
solve_params = {
    "h_int":             U * n("up", 0) * n("down", 0),
    "h_loc0":            -mu * (n("up", 0) + n("down", 0)),
    "n_cycles":          10000,
    "n_warmup_cycles":   1000,
    "length_cycle":      50,
    "random_seed":       123 * mpi.rank + 567,
    "random_name":       "",
    "measure_pert_order": True,
}

S.solve(**solve_params)

# Save output
if mpi.is_master_node():
    with HDFArchive("spin_spin.out.h5", 'w') as A:
        A["G_tau"] = S.G_tau
        A["perturbation_order"] = S.perturbation_order

# Compare against reference
if mpi.is_master_node():
    with HDFArchive("spin_spin.ref.h5", 'r') as A:
        assert_block_gfs_are_close(A["G_tau"], S.G_tau)
