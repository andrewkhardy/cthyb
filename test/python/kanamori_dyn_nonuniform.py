# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# Dynamical (retarded) Hubbard-Kanamori interaction with a genuinely NON-uniform coupling:
# same-spin pairs get Q_tau, opposite-spin pairs get a different curve, 0.5*Q_tau -- same
# static h_loc (full Kanamori, spin-flip + pair-hopping) as kanamori_dyn.py, but this time
# U(tau) != U'(tau), which is the physically realistic case (e.g. GW-derived screening
# generically gives different retarded couplings for same-spin vs. opposite-spin channels).
#
# This still goes in full to the analytic Lang-Firsov path (0 stochastic), and it's worth
# spelling out why that's not a coincidence: find_conserved_density_combinations finds
# N_up = sum_a n_{a,up} and N_down = sum_a n_{a,down} as the conserved combinations (see
# kanamori_dyn.py's docstring). As *vectors* over the 4 spin-orbitals, N_up and N_down have
# disjoint support (N_up is zero on every down-orbital and vice versa), so the Lang-Firsov
# part split_density_couplings may use is any coupling matrix that is constant on each spin
# block, with the same-spin and opposite-spin blocks entirely independent of each other.
# Here every same-spin entry is Q_tau and every opposite-spin entry is 0.5*Q_tau, so each
# block is already constant, the split takes all of it and the residual vanishes. Same-spin
# and opposite-spin couplings being different functions of tau is therefore not a special
# case needing separate code.
#
# The diagonal (Holstein) self-terms use U's curve (Q_tau) here: a same-spin self-term (a==a)
# sits in the same spin block as a same-spin cross-term (a!=b), so giving it a different curve
# would make that block non-constant, and whatever does not fit would be sampled stochastically.
#
# To regenerate kanamori_dyn_nonuniform.ref.h5:
#   1. Run this script once (produces kanamori_dyn_nonuniform.out.h5)
#   2. Verify the results are physically reasonable
#   3. Copy kanamori_dyn_nonuniform.out.h5 -> kanamori_dyn_nonuniform.ref.h5

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

# H_loc parameters (matches kanamori_dyn.py)
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

S = Solver(beta=beta, gf_struct=gf_struct, n_iw=1025, n_tau=2500, delta_interface=True)
S.Delta_tau << Fourier(delta_w)

# Non-uniform: same-spin pairs get Q_tau, opposite-spin pairs get a genuinely different
# curve, 0.5*Q_tau.
kanamori_dynamical_vertices(S, spin_names, list(range(n_orb)), U=Q_tau, Uprime=0.5 * Q_tau, spin_flip=False)
for s in spin_names:
    for a in range(n_orb):
        S.add_dyn_vertex(n(s, a), n(s, a), _as_scalar_gf(Q_tau))

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
    with HDFArchive("kanamori_dyn_nonuniform.out.h5", 'w') as A:
        A["G_tau"] = S.G_tau

if mpi.is_master_node():
    with HDFArchive("kanamori_dyn_nonuniform.ref.h5", 'r') as A:
        assert_block_gfs_are_close(A["G_tau"], S.G_tau)
        print("G_tau matches reference")
