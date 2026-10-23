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
#
# The comparison is statistical: with that sign, a serial run leaves ~0.1 of noise on every G_l, at
# every l. Each split is solved twice, with independent seeds and half the cycles, and compared through
# the lowest L_MAX coefficients of G_l averaged over the spins and the two equivalent sites (one on-site
# and one inter-site coefficient per l): they must agree with "max" within Z_MAX standard errors, the
# error of each run being the difference of its two halves, pooled over those coefficients. The same
# file run under mpirun on many ranks (each rank does the full cycles) is the high-statistics check.

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
L_MAX = 8      # G_l compared for l < L_MAX: beyond, this model's coefficients are below the noise
Z_MAX = 6.0    # agreement within Z_MAX standard errors
ABS_TOL = 5e-3 # plus this, for the Legendre fit of the Lang-Firsov part once the noise is that small

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


def symmetrized(S):
    """G_l for l < L_MAX averaged over the spins and the two equivalent sites: (L_MAX, 2), on-site and inter-site"""
    g = np.mean([S.G_l[s].data[:L_MAX].real for s in spin_names], axis=0)
    return np.stack([0.5 * (g[:, 0, 0] + g[:, 1, 1]), 0.5 * (g[:, 0, 1] + g[:, 1, 0])], axis=1)


def mean_and_error(halves):
    """The mean of the two halves of a split and its standard error, |a - b| / 2 pooled over the coefficients"""
    a, b = (symmetrized(S) for S in halves)
    return 0.5 * (a + b), np.sqrt(np.mean((0.5 * (a - b))**2))


S_static = build_and_solve("min", n_cycles=100000, seed_offset=0, retarded=False)
cycles = {"min": 200000, "mean": 200000, "max": 200000, "all stochastic": 300000}
runs = {name: [build_and_solve("min" if name == "all stochastic" else name, n_cycles // 2, seed_offset=1000 * (2 * k + h + 1),
                               lang_firsov=name != "all stochastic") for h in range(2)]
        for k, (name, n_cycles) in enumerate(cycles.items())}

if mpi.is_master_node():
    sign = {name: min(S.average_sign for S in halves) for name, halves in runs.items()}
    mpi.report("average sign (smaller half): static %.5f, " % S_static.average_sign + ", ".join("%s %.5f" % kv for kv in sign.items()))
    for name, s in sign.items():
        assert s > 0.0, "%s: non-positive average sign %s" % (name, s)
    assert sign["max"] >= MIN_SIGN, "max: average sign %s below %s" % (sign["max"], MIN_SIGN)
    reference, reference_error = mean_and_error(runs["max"])
    for name, halves in runs.items():
        if name == "max":
            continue
        if sign[name] < MIN_SIGN:
            mpi.report("%s: average sign %.5f below %s, G_l not compared" % (name, sign[name], MIN_SIGN))
            continue
        g, error = mean_and_error(halves)
        difference = np.abs(g - reference)
        tolerance = Z_MAX * np.hypot(error, reference_error) + ABS_TOL
        l, kind = np.unravel_index(difference.argmax(), difference.shape)
        mpi.report("%s - max: largest |dG_l| %.4f (l = %d, %s), standard error %.4f, tolerance %.4f"
                   % (name, difference.max(), l, ("on-site", "inter-site")[kind], np.hypot(error, reference_error), tolerance))
        assert difference.max() < tolerance, "%s and max G_l differ by %.4f at l = %d, more than %.4f" % (name, difference.max(), l, tolerance)
