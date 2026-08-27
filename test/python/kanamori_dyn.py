# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# Dynamical (retarded) Hubbard-Kanamori interaction. The static h_int carries the full
# Kanamori structure (U, U'=U-2J, J_hund, with both spin-flip and pair-hopping -- ordinary
# many_body_operator terms, no vertex-shape restriction applies to h_int). On top of that,
# a single boson couples uniformly to the total density across every spin-orbital: every
# off-diagonal pair via kanamori_dynamical_vertices (U=Uprime=Q_tau), *and* every diagonal
# (a==a) self-term explicitly via add_dyn_vertex, all sharing the same Q_tau coupling.
#
# The diagonal terms are not optional here: find_conserved_density_combinations finds that
# individual orbital densities do not commute with h_loc (real spin-flip/pair-hopping), but
# that total spin-up density and total spin-down density each individually do (equivalently,
# total charge and total S_z) -- and recover_conserved_density_groups only ever promotes a
# *completely* user-specified coupling matrix (diagonal entries included) to the analytic
# Lang-Firsov path, matching it exactly against those conserved combinations. Nothing is
# inferred: dropping the diagonal vertices below silently changes the physics being asked
# for (no Holstein self-term), so it correctly falls back to the stochastic path instead --
# see test/python/kanamori_dyn_selfconsistency.py's docstring for that case.
#
# To regenerate kanamori_dyn.ref.h5:
#   1. Run this script once (produces kanamori_dyn.out.h5)
#   2. Verify the results are physically reasonable
#   3. Copy kanamori_dyn.out.h5 -> kanamori_dyn.ref.h5

import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import *
from triqs.operators import n
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure
from h5 import HDFArchive
from triqs.utility.comparison_tests import *
from triqs_cthyb import Solver, kanamori_dynamical_vertices
from triqs_cthyb.dynamical_interactions import _as_scalar_gf
from itertools import product

# H_loc parameters (matches test/python/kanamori_py.py)
beta = 10.0
n_orb = 2
mu = 1.0
U = 2.0
J = 0.2

# Poles of Delta
epsilon = 2.3

# Retarded phonon kernel Q(tau), coupled uniformly to total density -- same closed form
# as benchmark/dynamic_int/multiorb_spin_spin.py / spin_spin.py's single bosonic mode.
omega_0 = 1.0
g = 0.5
n_tau_bosonic = 2001
tau_mesh_pts = np.linspace(0, beta, n_tau_bosonic)
Q_data = -(1.0 / (2.0 * omega_0)) * np.cosh(omega_0 * (tau_mesh_pts - beta / 2.0)) / np.sinh(omega_0 * beta / 2.0)
Q_tau = GfImTime(indices=[0], beta=beta, statistic='Boson', n_points=n_tau_bosonic)
Q_tau.data[:, 0, 0] = g**2 * Q_data

# Hybridization matrices (matches kanamori_py.py)
V = 1.0 * np.eye(n_orb) + 0.1 * (np.ones(n_orb) - np.eye(n_orb))
delta_w = GfImFreq(target_shape=(n_orb, n_orb), beta=beta)
delta_w << inverse(iOmega_n - epsilon) + inverse(iOmega_n + epsilon)
delta_w.from_L_G_R(V, delta_w, V)

# Block structure of GF -- kanamori_dynamical_vertices needs n(spin, orb) indexing, i.e.
# spin-named blocks each holding n_orb orbitals (off_diag=True), not the per-orbital-per-spin
# block convention used elsewhere in this repo (e.g. multiorb_spin_spin.py).
spin_names = ('up', 'down')
gf_struct = set_operator_structure(spin_names, n_orb, True)

# Static Hamiltonian: full Kanamori, including spin-flip and pair-hopping.
H_int = h_int_kanamori(spin_names, n_orb,
                        np.array([[0, U - 3 * J], [U - 3 * J, 0]]),      # same spin
                        np.array([[U, U - 2 * J], [U - 2 * J, U]]),      # opposite spin
                        J, spin_flip=True, pair_hopping=True, off_diag=True)

N = sum(n(s, a) for s, a in product(spin_names, range(n_orb)))

# Construct solver
S = Solver(beta=beta, gf_struct=gf_struct, n_iw=1025, n_tau=2500, delta_interface=True)
S.Delta_tau << Fourier(delta_w)

# Dynamical part: same coupling for U and Uprime (uniform over the whole group), no
# dynamical spin-flip channel in this example.
kanamori_dynamical_vertices(S, spin_names, list(range(n_orb)), U=Q_tau, Uprime=Q_tau, spin_flip=False)

# Diagonal (Holstein) self-terms, explicitly, with the same shared coupling -- required
# for the group to be *completely* specified (see module docstring above).
for s in spin_names:
    for a in range(n_orb):
        S.add_dyn_vertex(n(s, a), n(s, a), _as_scalar_gf(Q_tau))

# Solve parameters -- deterministic, small statistics (matches kanamori.cpp/kanamori_py.py).
# lang_firsov defaults to True; this test never touches the stochastic dynamical path since
# recover_conserved_density_groups resums the fully-specified uniform coupling analytically.
solve_params = {
    "h_int": H_int,
    "h_loc0": mu * N,
    "max_time": -1,
    "random_name": "",
    "random_seed": 123 * mpi.rank + 567,
    "length_cycle": 50,
    "n_warmup_cycles": 50,
    "n_cycles": 5000,
    "move_double": False,
    "measure_pert_order": True,
}

S.solve(**solve_params)

if mpi.is_master_node():
    mpi.report("Average sign = %s" % S.average_sign)
    with HDFArchive("kanamori_dyn.out.h5", 'w') as A:
        A["G_tau"] = S.G_tau

if mpi.is_master_node():
    with HDFArchive("kanamori_dyn.ref.h5", 'r') as A:
        assert_block_gfs_are_close(A["G_tau"], S.G_tau)
        print("G_tau matches reference")
