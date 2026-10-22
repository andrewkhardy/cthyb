# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# lang_firsov_split = "min", "mean" and "max" on a dimer whose densities do not commute with h_loc.
# Two sites (orbitals 0 and 1 of each spin block) are joined by a hopping t_perp in h_loc0, so only
# N_up and N_down are conserved and every retarded density vertex goes to split_density_couplings.
#
# The retarded coupling is the bilayer pattern: Q(tau) < 0 on-site and, between the sites,
# c(tau) = Q(tau) (0.3 - 0.6 sin^2(pi tau / beta)), which is negative near tau = 0, beta and positive
# around beta/2. The matrix [[Q, c], [c, Q]] stays negative semidefinite (|c| <= 0.3 |Q|), as for a
# causal screened interaction, but its off-diagonal entry changes sign. A stochastic vertex has
# weight -R(tau), so:
#   min   D_LF = 0 where the block has entries of both signs, so R keeps the positive c(tau):
#         negative weights around beta/2
#   mean  R = D - average has both signs everywhere: the most negative weight of the three
#   max   D_LF = the largest entry, R <= 0 everywhere: no negative weight from the retarded vertices
# All three splits are exact, so they and the all-stochastic run must give the same G_l. A static run
# (no retarded interaction) gives the sign of the dimer itself (t_perp in h_loc0, off-diagonal Delta),
# which none of the splits can remove. Runs with a sign below MIN_SIGN are reported, not compared.

import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import *
from triqs.operators import n, c, c_dag
from triqs.operators.util.op_struct import set_operator_structure
from triqs.utility.comparison_tests import *
from triqs_cthyb import Solver
from itertools import product

beta = 10.0
n_site = 2
mu = 1.0
U = 2.0
t_perp = -0.5
epsilon = 2.3
MIN_SIGN = 0.1

omega_0 = 1.0
g = 0.5
n_tau_bosonic = 2001
tau_mesh_pts = np.linspace(0, beta, n_tau_bosonic)
Q_data = -(1.0 / (2.0 * omega_0)) * np.cosh(omega_0 * (tau_mesh_pts - beta / 2.0)) / np.sinh(omega_0 * beta / 2.0)
Q_tau = GfImTime(indices=[0], beta=beta, statistic='Boson', n_points=n_tau_bosonic)
Q_tau.data[:, 0, 0] = g**2 * Q_data
C_tau = Q_tau.copy()
C_tau.data[:, 0, 0] = g**2 * Q_data * (0.3 - 0.6 * np.sin(np.pi * tau_mesh_pts / beta)**2)

V = 1.0 * np.eye(n_site) + 0.1 * (np.ones(n_site) - np.eye(n_site))
delta_w = GfImFreq(target_shape=(n_site, n_site), beta=beta)
delta_w << inverse(iOmega_n - epsilon) + inverse(iOmega_n + epsilon)
delta_w.from_L_G_R(V, delta_w, V)

spin_names = ('up', 'down')
gf_struct = set_operator_structure(spin_names, n_site, True)

H_int = sum(U * n('up', a) * n('down', a) for a in range(n_site))
N = sum(n(s, a) for s, a in product(spin_names, range(n_site)))
H_hop = sum(t_perp * (c_dag(s, 0) * c(s, 1) + c_dag(s, 1) * c(s, 0)) for s in spin_names)


def build_and_solve(split, n_cycles, seed_offset, lang_firsov=True, retarded=True):
    S = Solver(beta=beta, gf_struct=gf_struct, n_iw=1025, n_tau=2500, delta_interface=True)
    S.Delta_tau << Fourier(delta_w)
    if retarded:
        for (s1, a), (s2, b) in product(product(spin_names, range(n_site)), repeat=2):
            S.add_dyn_int(Q_tau if a == b else C_tau, n(s1, a), n(s2, b))
    S.solve(h_int=H_int, h_loc0=-mu * N + H_hop, max_time=-1, random_name="",
            random_seed=seed_offset + 123 * mpi.rank + 567,
            length_cycle=50, n_warmup_cycles=max(50, n_cycles // 20), n_cycles=n_cycles,
            move_double=True, lang_firsov=lang_firsov, lang_firsov_split=split, measure_G_l=True)
    return S


S_static = build_and_solve("min", n_cycles=100000, seed_offset=0, retarded=False)
runs = {split: build_and_solve(split, n_cycles=200000, seed_offset=1000 * (k + 1))
        for k, split in enumerate(("min", "mean", "max"))}
runs["all stochastic"] = build_and_solve("min", n_cycles=300000, seed_offset=4000, lang_firsov=False)

if mpi.is_master_node():
    mpi.report("average sign: static %.5f, " % S_static.average_sign
               + ", ".join("%s %.5f" % (k, S.average_sign) for k, S in runs.items()))
    for k, S in runs.items():
        assert S.average_sign > 0.0, "%s: non-positive average sign %s" % (k, S.average_sign)
    reference = runs["max"]
    assert reference.average_sign >= MIN_SIGN, "max: average sign %s below %s" % (reference.average_sign, MIN_SIGN)
    for k, S in runs.items():
        if k == "max":
            continue
        if S.average_sign < MIN_SIGN:
            mpi.report("%s: average sign %.5f below %s, G_l not compared" % (k, S.average_sign, MIN_SIGN))
            continue
        assert_block_gfs_are_close(S.G_l, reference.G_l, precision=0.1)
        mpi.report("%s and max G_l agree within tolerance" % k)
