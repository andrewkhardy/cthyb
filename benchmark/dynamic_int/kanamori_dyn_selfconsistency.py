# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# Production-scale version of test/python/kanamori_dyn_selfconsistency.py: solves the
# same dynamical Hubbard-Kanamori setup as kanamori_dyn.py (off-diagonal vertices from
# kanamori_dynamical_vertices *plus* explicit diagonal self-terms -- see kanamori_dyn.py's
# module docstring for why the diagonal is required) twice -- once with lang_firsov=True
# (analytic, via recover_conserved_density_groups) and once with lang_firsov=False
# (forced stochastic double expansion) -- and saves both results so they can be compared
# (e.g. via G_l, which converges much faster than raw G_tau; see the module docstring in
# test/python/kanamori_dyn_selfconsistency.py for why).
#
# Two separate Solver instances are used rather than re-solving one solver twice:
# solve() mutates h_loc in place via the Lang-Firsov K'(0) shift, so reusing one
# instance across both runs would contaminate the second.

import argparse
import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import *
from triqs.operators import n
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure
import h5
from triqs_cthyb import Solver, kanamori_dynamical_vertices
from triqs_cthyb.dynamical_interactions import _as_scalar_gf
from itertools import product

parser = argparse.ArgumentParser(description='Lang-Firsov vs. forced-stochastic self-consistency check for the dynamical Hubbard-Kanamori interaction.')
parser.add_argument('--n_orb', type=int, default=2, help='Number of orbitals')
parser.add_argument('--U', type=float, default=2.0, help='Intra-orbital Hubbard U')
parser.add_argument('--Jhund', type=float, default=0.2, help='Hund coupling J (static: spin-flip + pair-hopping; Uprime = U - 2*Jhund)')
parser.add_argument('--beta', type=float, default=10.0, help='Inverse temperature')
parser.add_argument('--omega_0', type=float, default=1.0, help='Phonon/bosonic mode frequency')
parser.add_argument('--g', type=float, default=0.5, help='Electron-phonon coupling strength (coupling to total density is g**2 * Q(tau))')
parser.add_argument('--epsilon', type=float, default=2.3, help='Bath pole position (two-pole analytic bath)')
parser.add_argument('--n_cycles_lf', type=int, default=200000, help='MC cycles for the lang_firsov=True (analytic) run')
parser.add_argument('--n_cycles_stoch', type=int, default=300000, help='MC cycles for the forced lang_firsov=False (stochastic) run')
parser.add_argument('--n_warmup_frac', type=float, default=0.05, help='Warmup cycles as a fraction of n_cycles for each run')
parser.add_argument('--length_cycle', type=int, default=50, help='Length of MC cycle')
parser.add_argument('--dyn_n_l', type=int, default=50, help='Number of Legendre polynomials for dynamical interactions')
args, unknown = parser.parse_known_args()

n_orb = args.n_orb
U = args.U
J = args.Jhund
beta = args.beta
omega_0 = args.omega_0
g = args.g
epsilon = args.epsilon
mu = 1.0

spin_names = ('up', 'down')
gf_struct = set_operator_structure(spin_names, n_orb, True)

n_tau_bosonic = 2001
tau_mesh_pts = np.linspace(0, beta, n_tau_bosonic)
Q_data = -(1.0 / (2.0 * omega_0)) * np.cosh(omega_0 * (tau_mesh_pts - beta / 2.0)) / np.sinh(omega_0 * beta / 2.0)
Q_tau = GfImTime(indices=[0], beta=beta, statistic='Boson', n_points=n_tau_bosonic)
Q_tau.data[:, 0, 0] = g**2 * Q_data

n_iw = 1025
n_tau = 2500
V = np.eye(n_orb)
delta_w = GfImFreq(target_shape=(n_orb, n_orb), beta=beta, n_points=n_iw)
delta_w << inverse(iOmega_n - epsilon) + inverse(iOmega_n + epsilon)
delta_w.from_L_G_R(V, delta_w, V)

H_int = h_int_kanamori(spin_names, n_orb,
                        np.array([[0 if a1 == a2 else U - 3 * J for a2 in range(n_orb)] for a1 in range(n_orb)]),
                        np.array([[U if a1 == a2 else U - 2 * J for a2 in range(n_orb)] for a1 in range(n_orb)]),
                        J, spin_flip=True, pair_hopping=True, off_diag=True)

N = sum(n(s, a) for s, a in product(spin_names, range(n_orb)))


def build_and_solve(lang_firsov, n_cycles, seed_offset):
    S = Solver(beta=beta, gf_struct=gf_struct, n_iw=n_iw, n_tau=n_tau, delta_interface=True)
    S.Delta_tau << Fourier(delta_w)
    kanamori_dynamical_vertices(S, spin_names, list(range(n_orb)), U=Q_tau, Uprime=Q_tau, spin_flip=False)
    for s in spin_names:
        for a in range(n_orb):
            S.add_dyn_vertex(n(s, a), n(s, a), _as_scalar_gf(0.5 * Q_tau))
    S.solve(h_int=H_int, h_loc0=mu * N,
            random_seed=seed_offset + 123 * mpi.rank + 567, random_name="",
            length_cycle=args.length_cycle,
            n_warmup_cycles=max(50, int(n_cycles * args.n_warmup_frac)),
            n_cycles=n_cycles, lang_firsov=lang_firsov,
            measure_G_l=True, dyn_n_l=args.dyn_n_l, measure_pert_order=True)
    return S


mpi.report("=== Lang-Firsov run (n_cycles=%d) ===" % args.n_cycles_lf)
S_lf = build_and_solve(lang_firsov=True, n_cycles=args.n_cycles_lf, seed_offset=0)
mpi.report("LF average sign = %s" % S_lf.average_sign)

mpi.report("=== Forced-stochastic run (n_cycles=%d) ===" % args.n_cycles_stoch)
S_stoch = build_and_solve(lang_firsov=False, n_cycles=args.n_cycles_stoch, seed_offset=1000)
mpi.report("Stochastic average sign = %s" % S_stoch.average_sign)

if mpi.is_master_node():
    for bl in S_lf.G_tau.indices:
        gl_diff = np.max(np.abs(S_lf.G_l[bl].data[:, 0, 0] - S_stoch.G_l[bl].data[:, 0, 0]))
        print("block %s: max|G_l_lf - G_l_stoch| = %s" % (bl, gl_diff))

    filename = (f"/mnt/home/ahardy/ceph/CTHYB_Data/kanamori_dyn_selfconsistency_norb-{n_orb}_U-{U}_J-{J}"
                f"_g-{g}_w0-{omega_0}_beta-{beta}_nclf-{args.n_cycles_lf}_ncstoch-{args.n_cycles_stoch}.h5")
    with h5.HDFArchive(filename, "w") as A:
        A['G_tau_lf'] = S_lf.G_tau
        A['G_l_lf'] = S_lf.G_l
        A['average_sign_lf'] = S_lf.average_sign
        A['G_tau_stoch'] = S_stoch.G_tau
        A['G_l_stoch'] = S_stoch.G_l
        A['average_sign_stoch'] = S_stoch.average_sign
        A['n_orb'] = n_orb
        A['U'] = U
        A['J'] = J
        A['g'] = g
        A['omega_0'] = omega_0
        A['beta'] = beta
        A['n_cycles_lf'] = args.n_cycles_lf
        A['n_cycles_stoch'] = args.n_cycles_stoch
    print(f"Results saved to {filename}")
