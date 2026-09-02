# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# Dynamical (retarded) Hubbard-Kanamori interaction with a genuine dynamical spin-flip
# channel -- production-scale benchmark, structurally kanamori_dyn.py plus
# kanamori_dynamical_vertices(..., spin_flip=True, J_hund=...) turned on.
#
# Static h_int carries the full Kanamori structure (U, U'=U-2J, J_hund, spin-flip +
# pair-hopping), exactly as in kanamori_dyn.py. The dynamical part now has two pieces:
#   1. The same uniform density-density phonon coupling as kanamori_dyn.py (every
#      off-diagonal pair via kanamori_dynamical_vertices(U=Uprime=Q_tau), every
#      diagonal self-term explicit) -- completely specified, so it is still fully
#      recovered to the analytic Lang-Firsov path when lang_firsov=True.
#   2. A genuine *dynamical* (retarded) inter-orbital spin-flip coupling,
#      kanamori_dynamical_vertices(..., spin_flip=True, J_hund=Jsf_tau): each such
#      vertex is c_dag(s1,a1)*c(s2,a1) (tau) * c_dag(s2,a2)*c(s1,a2) (0), a1 != a2,
#      s1 != s2 -- not a density bilinear, so classify_dyn_vertices routes it straight
#      to the stochastic double expansion (moves/insert_dyn.cpp/remove_dyn.cpp)
#      regardless of lang_firsov, unlike every other dynamical vertex in the
#      kanamori_dyn* family so far. This is the first production run exercising the
#      stochastic path for a genuinely multi-orbital dynamical Jperp-type vertex (the
#      only prior stochastic-path test, spin_spin.cpp/.py, is single-orbital and uses
#      the separate scalar Jperp_tau convenience path, not add_dyn_vertex).
#
# Expect S.perturbation_order_dyn to be populated (nonzero dynamical-vertex insertions)
# even with lang_firsov=True, since the spin-flip vertices can never be promoted to the
# analytic path -- unlike kanamori_dyn.py, where a fully-specified run shows 0
# stochastic. The mu correction (see kanamori_dyn.py) only involves the density part:
# apply_lang_firsov_shift only ever sees vertices classify_dyn_vertices accepted, so
# lang_firsov_U_renorm/lang_firsov_mu_renorm are unaffected by the spin-flip channel.

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

parser = argparse.ArgumentParser(description='Run dynamical Hubbard-Kanamori benchmarking with a dynamical spin-flip channel.')
parser.add_argument('--n_orb', type=int, default=2, help='Number of orbitals')
parser.add_argument('--U', type=float, default=2.0, help='Intra-orbital Hubbard U')
parser.add_argument('--Jhund', type=float, default=0.2, help='Hund coupling J (static: spin-flip + pair-hopping; Uprime = U - 2*Jhund)')
parser.add_argument('--beta', type=float, default=10.0, help='Inverse temperature')
parser.add_argument('--omega_0', type=float, default=1.0, help='Shared boson frequency for both the density and spin-flip dynamical channels')
parser.add_argument('--g', type=float, default=0.5, help='Electron-phonon coupling to total density (coupling is g**2 * Q(tau))')
parser.add_argument('--g_sf', type=float, default=0.5, help='Electron-phonon coupling to the inter-orbital spin-flip channel (coupling is g_sf**2 * Q(tau)); this is the piece that forces vertices onto the stochastic path. Short local scan at beta=10, U=2, J=0.2: g_sf=0.3 gives average sign 0.97 but only ~2%% of measures carry a spin-flip vertex; 0.5 gives 0.89/~7%%; 0.7 gives 0.86/~11%%; 1.0 gives 0.75/~26%%')
parser.add_argument('--half_bandwidth', type=float, default=2.0, help='Half-bandwidth D of the Bethe-lattice (semicircular) bath, per orbital')
parser.add_argument('--n_cycles', type=int, default=1000000, help='Number of MC cycles')
parser.add_argument('--n_warmup_cycles', type=int, default=50000, help='Warmup cycles')
parser.add_argument('--length_cycle', type=int, default=100, help='Length of MC cycle')
parser.add_argument('--dyn_n_l', type=int, default=100, help='Number of Legendre polynomials for dynamical interactions')
parser.add_argument("--lang_firsov", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True, help="Whether to route eligible (density) vertices through Lang-Firsov (False forces everything, including the density part, through the stochastic double expansion)")
args, unknown = parser.parse_known_args()

n_orb = args.n_orb
U = args.U
J = args.Jhund
beta = args.beta
omega_0 = args.omega_0
g = args.g
g_sf = args.g_sf
half_bandwidth = args.half_bandwidth
# Half filling for h_int_kanamori's density-density matrix alone (see kanamori_dyn.py);
# corrected below for the density dynamical channel's own static contribution. The
# spin-flip channel never touches apply_lang_firsov_shift (it's never Lang-Firsov
# eligible), so it does not enter this formula or the correction below.
mu_bare = 0.5 * U + (n_orb - 1) * 0.5 * (U - 2 * J) + (n_orb - 1) * 0.5 * (U - 3 * J)

spin_names = ('up', 'down')
gf_struct = set_operator_structure(spin_names, n_orb, True)

# Shared retarded boson kernel Q(tau) -- same closed form used throughout
# benchmark/dynamic_int; the density and spin-flip channels use independent couplings
# (g, g_sf) to the same underlying mode.
n_tau_bosonic = 2001
tau_mesh_pts = np.linspace(0, beta, n_tau_bosonic)
Q_data = -(1.0 / (2.0 * omega_0)) * np.cosh(omega_0 * (tau_mesh_pts - beta / 2.0)) / np.sinh(omega_0 * beta / 2.0)
Q_tau = GfImTime(indices=[0], beta=beta, statistic='Boson', n_points=n_tau_bosonic)
Q_tau.data[:, 0, 0] = g**2 * Q_data
Jsf_tau = GfImTime(indices=[0], beta=beta, statistic='Boson', n_points=n_tau_bosonic)
Jsf_tau.data[:, 0, 0] = g_sf**2 * Q_data

# Hybridization: decoupled Bethe-lattice (semicircular) bath, same per orbital.
n_iw = 1025
n_tau = 2500
g0_bethe = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
g0_bethe << SemiCircular(half_bandwidth)
delta_w = GfImFreq(target_shape=(n_orb, n_orb), beta=beta, n_points=n_iw)
for a in range(n_orb):
    delta_w[a, a] << (half_bandwidth / 2.0) ** 2 * g0_bethe[0, 0]

# Static Hamiltonian: full Kanamori, including spin-flip and pair-hopping.
H_int = h_int_kanamori(spin_names, n_orb,
                        np.array([[0 if a1 == a2 else U - 3 * J for a2 in range(n_orb)] for a1 in range(n_orb)]),
                        np.array([[U if a1 == a2 else U - 2 * J for a2 in range(n_orb)] for a1 in range(n_orb)]),
                        J, spin_flip=True, pair_hopping=True, off_diag=True)

# Spin-orbitals in the same order as gf_struct/linindex, so a per-orbital mu list lines
# up positionally with S.lang_firsov_mu_renorm / S.lang_firsov_U_renorm below.
spin_orbitals = list(product(spin_names, range(n_orb)))
n_ops = [n(s, a) for s, a in spin_orbitals]

# Construct solver
S = Solver(beta=beta, gf_struct=gf_struct, n_iw=n_iw, n_tau=n_tau, delta_interface=True)
S.Delta_tau << Fourier(delta_w)

# Dynamical part 1: uniform density-density coupling, completely specified (off-diagonal
# + diagonal self-terms) -- recovers exactly to the analytic Lang-Firsov path, as in
# kanamori_dyn.py.
kanamori_dynamical_vertices(S, spin_names, list(range(n_orb)), U=Q_tau, Uprime=Q_tau, spin_flip=False)
for s in spin_names:
    for a in range(n_orb):
        S.add_dyn_vertex(n(s, a), n(s, a), _as_scalar_gf(Q_tau))

# Dynamical part 2: genuine dynamical spin-flip coupling -- never Lang-Firsov eligible,
# always goes through the stochastic double expansion (see module docstring above).
if n_orb > 1 and g_sf != 0.0:
    kanamori_dynamical_vertices(S, spin_names, list(range(n_orb)), spin_flip=True, J_hund=Jsf_tau)

solve_params = {
    "h_int": H_int,
    "length_cycle": args.length_cycle,
    "n_warmup_cycles": args.n_warmup_cycles,
    "n_cycles": args.n_cycles,
    "measure_pert_order": True,
    "lang_firsov": args.lang_firsov,
    "dyn_n_l": args.dyn_n_l,
}

# Mu correction (see kanamori_dyn.py for the derivation): a cheap probe solve exposes
# how much the density dynamical channel's static part shifts the half-filling mu.
# Only the density channel enters lang_firsov_U_renorm/lang_firsov_mu_renorm -- the
# spin-flip channel is never folded into h_loc, so it is correctly absent here.
probe_params = dict(solve_params, n_cycles=1, n_warmup_cycles=1)
S.solve(**probe_params, h_loc0=-mu_bare * sum(n_ops))
mu = [mu_bare] * len(spin_orbitals)
if len(S.lang_firsov_mu_renorm) > 0:
    U_renorm = S.lang_firsov_U_renorm
    mu_renorm = S.lang_firsov_mu_renorm
    for i in range(len(spin_orbitals)):
        mu_half_correct_i = 0.5 * sum(U_renorm[i][j] for j in range(len(spin_orbitals)) if j != i)
        mu[i] = mu_bare + (mu_half_correct_i - mu_renorm[i])
    if mpi.is_master_node():
        print(f"Lang-Firsov mu correction: mu_bare={mu_bare}, corrected mu={mu}")

# Solve (real production run, with the corrected mu)
S.solve(**solve_params, h_loc0=-sum(mu[i] * n_ops[i] for i in range(len(spin_orbitals))))

if mpi.is_master_node():
    # The solve() log above already reports the definitive split, e.g.
    #   "Dynamical interaction vertices: 16 analytic (Lang-Firsov), 4 stochastic"
    # -- for n_orb=2 the 4 stochastic ones are exactly the dynamical spin-flip vertices.
    print(f"perturbation_order_dyn measured: {S.perturbation_order_dyn is not None}")

# Save data
if mpi.is_master_node():
    filename = (f"/mnt/home/ahardy/ceph/CTHYB_Data/kanamori_dyn_spinflip_cthyb_norb-{n_orb}_U-{U}_J-{J}"
                f"_g-{g}_gsf-{g_sf}_w0-{omega_0}_beta-{beta}_nc-{args.n_cycles}_lf={args.lang_firsov}.h5")
    with h5.HDFArchive(filename, "w") as A:
        A['G_tau'] = S.G_tau
        A['perturbation_order'] = S.perturbation_order
        if S.perturbation_order_dyn is not None:
            A['perturbation_order_dynamical'] = S.perturbation_order_dyn
        A['average_sign'] = S.average_sign
        A['K_n'] = S.K_n
        A['n_orb'] = n_orb
        A['U'] = U
        A['J'] = J
        A['g'] = g
        A['g_sf'] = g_sf
        A['omega_0'] = omega_0
        A['beta'] = beta
        A['lang_firsov'] = args.lang_firsov
        A['mu_bare'] = mu_bare
        A['mu'] = mu
    print(f"Results saved to {filename}")
    print(f"Average sign = {S.average_sign}")
