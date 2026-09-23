# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.

# Single-orbital test with dynamical spin-spin interactions (D0_tau + Jperp_tau).
# Uses a simple analytical bath: Delta(iw) = 1/(iw - eps) + 1/(iw + eps)
# and a bosonic propagator: J(iw) = 4*l^2*w0/(iw^2 - w0^2)
#
# This mirrors the C++ spin_spin test but exercises the Python interface.
#
# This is a BIT-REPRODUCIBILITY test, not a statistical one. It compares a stochastic
# G_tau against a stored reference at the default precision of 1e-6, which only passes
# because the seed is fixed and the test runs serially (mpi.rank == 0, so the seed is
# exactly 567). It is sensitive by design -- it catches any unintended change to the
# solver -- but it also fails on every *intended* change to the random-number consumption
# pattern, such as adding a Monte Carlo move. That is not a physics regression, and the
# check below tells the two apart.
#
# For scale: n_tau = 10001 bins over 200000 cycles is ~20 samples per bin, so the per-bin
# noise is large and the max over 10001 bins is larger still. Measured 2026-09-18, two runs
# of identical code differing only by seed disagreed by max|dG| = 0.36, which is *more*
# than either disagreed with the stored reference. So a failure of order 0.1-0.4 here is
# consistent with pure Monte Carlo noise and says nothing about correctness on its own.
#
# To regenerate spin_spin.ref.h5:
#   1. Run this script once (produces spin_spin.out.h5)
#   2. Verify the results are physically reasonable. Do NOT just eyeball G(tau) -- at this
#      statistics level it is visibly noisy either way. Check instead:
#        - G(0) + G(beta) == -1 exactly, per spin (an identity, not a statistical statement)
#        - -G(beta) ~ 0.5 per spin, i.e. half filling
#        - Im Sigma(iw_n) < 0 at low frequency (causality)
#        - Re Sigma(iw_n) ~ mu = U/2 = 1.0 away from the lowest frequencies. This is exact
#          at half filling: G0 is built with the bare mu while the solver shifts internally
#          to mu_eff = U_eff/2, and the two offsets cancel. It is independent of the
#          reference, so it is the most informative single check here.
#      To decide whether a failure is noise or a real change, re-run with a different seed:
#      if seed-to-seed scatter is comparable to the disagreement with the reference, it is
#      noise.
#   3. Copy spin_spin.out.h5 -> spin_spin.ref.h5
#
# Last regenerated 2026-09-18, after the Bug A/C fixes added the swap_dyn move, which
# changed the random stream. Verified noise-only by the seed comparison described above.

import triqs.utility.mpi as mpi
from triqs.gf import *
from triqs.operators import n
from h5 import HDFArchive
from triqs.utility.comparison_tests import *
from triqs_cthyb import Solver

# Physical parameters (same as C++ test)
beta    = 10.0
U       = 2.0
mu      = U / 2.0   # half-filling
epsilon = 0.3        # bath level
l       = 0.5        # electron-boson coupling (weak enough for reasonable sign)
w0      = 1.0        # screening frequency

# Discretization
n_iw          = 1025
n_tau         = 10001
n_tau_bosonic = 10001

# gf_struct — note 'down' before 'up' to test block ordering independence
gf_struct = [('down', 1), ('up', 1)]

# Construct solver with Delta_tau interface
S = Solver(beta=beta, gf_struct=gf_struct, n_iw=n_iw, n_tau=n_tau,
           n_tau_bosonic=n_tau_bosonic, delta_interface=True)

# Hybridization: Bethe lattice starting guess Delta(iw) = t^2 * G(iw)
half_bandwidth = 1.0
t = half_bandwidth / 2.0
G_iw_init = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
G_iw_init << SemiCircular(half_bandwidth)
Delta_iw = G_iw_init.copy()
Delta_iw << t**2 * G_iw_init
S.Delta_tau << Fourier(Delta_iw)

# Bosonic propagators: J(iw) = 4*l^2*w0/(iw^2 - w0^2), D(iw) = l^2*w0/(iw^2 - w0^2)
# Built via mesh iteration since lazy expressions don't support this form
J0_iw = GfImFreq(indices=[0], beta=beta, n_points=n_iw, statistic='Boson')
D0_iw = GfImFreq(indices=[0], beta=beta, n_points=n_iw, statistic='Boson')
for iw in J0_iw.mesh:
    w = complex(iw)
    J0_iw[iw] = 4 * l**2 * w0 / (w**2 - w0**2)
    D0_iw[iw] = l**2 * w0 / (w**2 - w0**2)
J0_tau = GfImTime(indices=[0], beta=beta, n_points=n_tau_bosonic, statistic='Boson')
D0_tau = GfImTime(indices=[0], beta=beta, n_points=n_tau_bosonic, statistic='Boson')
J0_tau << Fourier(J0_iw)
D0_tau << Fourier(D0_iw)

# Jperp: spin-flip interaction (scalar gf at this interface)
S.Jperp_tau << U/4* J0_tau

# D0: density-density retarded interaction
# Sz*Sz decomposition: same-spin = +D0, opposite-spin = -D0
S.D0_tau["up", "up"]     << U/4* D0_tau
S.D0_tau["down", "down"] << U/4* D0_tau
S.D0_tau["up", "down"]   << -1.0 * U/4* D0_tau
S.D0_tau["down", "up"]   << -1.0 * U/4*  D0_tau

# Solve parameters — fixed seed for reproducibility
solve_params = {
    "h_int":             U * n("up", 0) * n("down", 0),
    "h_loc0":            -mu * (n("up", 0) + n("down", 0)),
    "n_cycles":          200000,
    "n_warmup_cycles":   20000,
    "length_cycle":      75,
    "random_seed":       123 * mpi.rank + 567,
    "random_name":       "",
    # The local proposal of insert_dyn (on by default) changes the random stream; kept off so
    # this reference keeps checking the uniform proposal bit for bit. The local one is
    # validated statistically by dyn_local_moves.py and kanamori_dyn_selfconsistency.py.
    "move_dyn_local":    False,
    "measure_pert_order": True,
    "perform_tail_fit": True,
    "fit_max_moment": 3,
    "fit_min_w": 1.2,
    "fit_max_w": 3.0
}

S.solve(**solve_params)

# Save output
if mpi.is_master_node():
    with HDFArchive("spin_spin.out.h5", 'w') as A:
        A["G_tau"] = S.G_tau
        A["perturbation_order"] = S.perturbation_order
        A["Delta_tau"] = S.Delta_tau
        A["G_iw"] = S.G_iw
        A["G_iw_raw"] = S.G_iw_raw
        A["Sigma_iw"] = S.Sigma_iw
        A["Sigma_iw_raw"] = S.Sigma_iw_raw
# Compare against reference
if mpi.is_master_node():
    with HDFArchive("spin_spin.ref.h5", 'r') as A:
        assert_block_gfs_are_close(A["G_tau"], S.G_tau)
        print("G_tau matches reference")
