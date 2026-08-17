# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# Physics validation for the dynamical Hubbard-Kanamori conserved-density recovery (see
# kanamori_dyn.py for the physical setup and find_conserved_density_combinations /
# recover_conserved_density_groups in dynamical_interactions.cpp for the mechanism). No
# external reference exists for this case (ctseg/CTINT can't do multi-orbital dynamical
# interactions), so the only correctness check available is internal: solve the identical
# physical setup (off-diagonal vertices *and* the explicit diagonal self-terms, together a
# completely specified coupling to total spin-up/spin-down density) twice -- once with
# lang_firsov=True (recovers every vertex to the analytic path: individual orbital
# densities do not commute with h_loc under real spin-flip/pair-hopping, but total
# spin-up and total spin-down density each do) and once with lang_firsov=False (forces
# every vertex through the stochastic insert_dyn/remove_dyn path instead) -- and check the
# two independent methods agree.
#
# Two separate Solver instances are used rather than re-solving one solver twice: solve()
# mutates h_loc in place via the Lang-Firsov K'(0) shift, so reusing one instance across
# both runs would contaminate the second.
#
# Comparison uses G_l (Legendre coefficients), not raw G_tau: the raw binned G_tau
# estimator has large single-bin noise at this vertex count/statistics level (isolated
# bins can be off by O(1) even though the underlying physics agrees -- verified by hand
# this session: max|G_tau_lf - G_tau_stoch| shrinks the same way max|G_l_lf - G_l_stoch|
# does as statistics increase, confirming it is noise, not a discrepancy). G_l is a
# smooth, well-regularized representation and converges much faster.

import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import *
from triqs.operators import n
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure
from triqs.utility.comparison_tests import *
from triqs_cthyb import Solver, kanamori_dynamical_vertices
from triqs_cthyb.dynamical_interactions import _as_scalar_gf
from itertools import product

# H_loc parameters -- identical physical setup to kanamori_dyn.py.
beta = 10.0
n_orb = 2
mu = 1.0
U = 2.0
J = 0.2
epsilon = 2.3

omega_0 = 1.0
g = 0.5
n_tau_bosonic = 2001
tau_mesh_pts = np.linspace(0, beta, n_tau_bosonic)
Q_data = -(1.0 / (2.0 * omega_0)) * np.cosh(omega_0 * (tau_mesh_pts - beta / 2.0)) / np.sinh(omega_0 * beta / 2.0)
Q_tau = GfImTime(indices=[0], beta=beta, statistic='Boson', n_points=n_tau_bosonic)
Q_tau.data[:, 0, 0] = g**2 * Q_data

V = 1.0 * np.eye(n_orb) + 0.1 * (np.ones(n_orb) - np.eye(n_orb))
delta_w = GfImFreq(target_shape=(n_orb, n_orb), beta=beta)
delta_w << inverse(iOmega_n - epsilon) + inverse(iOmega_n + epsilon)
delta_w.from_L_G_R(V, delta_w, V)

spin_names = ('up', 'down')
gf_struct = set_operator_structure(spin_names, n_orb, True)

H_int = h_int_kanamori(spin_names, n_orb,
                        np.array([[0, U - 3 * J], [U - 3 * J, 0]]),      # same spin
                        np.array([[U, U - 2 * J], [U - 2 * J, U]]),      # opposite spin
                        J, spin_flip=True, pair_hopping=True, off_diag=True)

N = sum(n(s, a) for s, a in product(spin_names, range(n_orb)))


def build_and_solve(lang_firsov, n_cycles, seed_offset):
    S = Solver(beta=beta, gf_struct=gf_struct, n_iw=1025, n_tau=2500, delta_interface=True)
    S.Delta_tau << Fourier(delta_w)
    kanamori_dynamical_vertices(S, spin_names, list(range(n_orb)), U=Q_tau, Uprime=Q_tau, spin_flip=False)
    for s in spin_names:
        for a in range(n_orb):
            S.add_dyn_vertex(n(s, a), n(s, a), _as_scalar_gf(0.5 * Q_tau))
    S.solve(h_int=H_int, h_loc0=mu * N, max_time=-1, random_name="",
            random_seed=seed_offset + 123 * mpi.rank + 567,
            length_cycle=50, n_warmup_cycles=max(50, n_cycles // 20), n_cycles=n_cycles,
            move_double=False, lang_firsov=lang_firsov, measure_G_l=True)
    return S


# Lang-Firsov run: fast (analytic), forced-stochastic run: needs more statistics to
# converge to the same precision (empirically confirmed this session -- average sign
# stays close to 1 in both, no severe sign problem for this particular setup, unlike the
# unresolved pure-stochastic sign problem documented for the single-orbital spin_spin
# case; see MULTIORBITAL_IMPLEMENTATION.md).
S_lf = build_and_solve(lang_firsov=True, n_cycles=200000, seed_offset=0)
S_stoch = build_and_solve(lang_firsov=False, n_cycles=300000, seed_offset=1000)

if mpi.is_master_node():
    mpi.report("LF average sign = %s, stochastic average sign = %s" % (S_lf.average_sign, S_stoch.average_sign))
    assert S_lf.average_sign > 0.9, "Unexpected sign problem in the Lang-Firsov run"
    assert S_stoch.average_sign > 0.9, "Unexpected sign problem in the forced-stochastic run"

    assert_block_gfs_are_close(S_lf.G_l, S_stoch.G_l, precision=0.08)
    mpi.report("Lang-Firsov and forced-stochastic G_l agree within tolerance")
