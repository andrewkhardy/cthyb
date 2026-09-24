# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.

# Statistical check of the global up <-> down swap with stochastic dynamical vertices present
# (move_global_full, moves/global.cpp): every operator is swapped at once, and the Jperp
# vertices must be carried over whole (S+S- <-> S-S+).
#
# The swap is a symmetry of this model, so it changes the proposal only, never the stationary
# distribution: the two runs below must agree within statistics. A bookkeeping error (a vertex
# left pointing at the old spin) shows up as a shift in G_l, or as up and down disagreeing.
#
# Model: the single-orbital spin_spin.py test (Jperp through the stochastic path, Sz.Sz
# through Lang-Firsov, so the production combination), moved off half filling so that the
# density is not pinned by particle-hole symmetry and a bias would show up in it.

import numpy as np
import triqs.utility.mpi as mpi
from triqs.gf import *
from triqs.operators import n
from triqs.utility.comparison_tests import *
from triqs_cthyb import Solver

beta, U, l, w0 = 10.0, 2.0, 0.5, 1.0
mu = U / 2.0 + 0.4   # off half filling
n_iw, n_tau, n_tau_bosonic = 1025, 2001, 2001
gf_struct = [('down', 1), ('up', 1)]

G_iw_init = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
G_iw_init << SemiCircular(1.0)
Delta_iw = G_iw_init.copy()
Delta_iw << 0.25 * G_iw_init

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


def solve(seed_offset, **moves):
    S = Solver(beta=beta, gf_struct=gf_struct, n_iw=n_iw, n_tau=n_tau, n_l=30,
               n_tau_bosonic=n_tau_bosonic, delta_interface=True)
    S.Delta_tau << Fourier(Delta_iw)
    S.Jperp_tau << U / 4 * J0_tau
    for s1 in ("up", "down"):
        for s2 in ("up", "down"):
            S.D0_tau[s1, s2] << (1.0 if s1 == s2 else -1.0) * U / 4 * D0_tau
    S.solve(h_int=U * n("up", 0) * n("down", 0), h_loc0=-mu * (n("up", 0) + n("down", 0)),
            n_cycles=200000, n_warmup_cycles=10000, length_cycle=50,
            random_name="", random_seed=seed_offset + 123 * mpi.rank + 567,
            move_double=False, measure_G_l=True, **moves)
    return S


spin_flip = {"spin_flip": {("up", 0): ("down", 0), ("down", 0): ("up", 0)}}

S_ref = solve(0)
S_flip = solve(2000, move_global=spin_flip, move_global_full=True, move_global_prob=0.1)

if mpi.is_master_node():
    def density(S):
        return {bl: -S.G_tau[bl].data[-1, 0, 0].real for bl in ("up", "down")}

    for name, S in (("reference", S_ref), ("spin flip", S_flip)):
        mpi.report(f"{name:10s} sign = {S.average_sign:.4f}  density = {density(S)}")

    assert_block_gfs_are_close(S_ref.G_l, S_flip.G_l, precision=0.05)
    # Paramagnetic model: up and down must agree in both runs, and the spin flip enforces it
    for S in (S_ref, S_flip):
        assert_gfs_are_close(S.G_l["up"], S.G_l["down"], precision=0.05)
    mpi.report("The spin-flip move agrees with the original move set within statistics")
