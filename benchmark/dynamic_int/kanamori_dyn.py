# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# Dynamical (retarded) Hubbard-Kanamori interaction, production-scale benchmark version
# of test/python/kanamori_dyn.py. Static h_int carries the full Kanamori structure (U,
# U'=U-2J, J_hund, with both spin-flip and pair-hopping). The dynamical part is a single
# boson/phonon coupled uniformly to total density across every spin-orbital -- every
# off-diagonal pair via kanamori_dynamical_vertices (U=Uprime=Q_tau), *and* every
# diagonal (a==a) self-term explicitly via add_dyn_vertex, all sharing the same Q_tau
# coupling. The diagonal terms are required, not optional: recover_conserved_density_groups
# only ever promotes a *completely* user-specified coupling matrix (diagonal included) to
# the analytic Lang-Firsov path (lang_firsov=True, the default) -- dropping them changes
# the physics being asked for (no Holstein self-term) and correctly falls back to the
# stochastic double expansion instead. Pass --lang_firsov False to force the stochastic
# path even with the diagonal present, for comparison (see kanamori_dyn_selfconsistency.py
# for a scripted version of that comparison).

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

parser = argparse.ArgumentParser(description='Run dynamical Hubbard-Kanamori benchmarking.')
parser.add_argument('--n_orb', type=int, default=2, help='Number of orbitals')
parser.add_argument('--U', type=float, default=2.0, help='Intra-orbital Hubbard U')
parser.add_argument('--Jhund', type=float, default=0.2, help='Hund coupling J (static: spin-flip + pair-hopping; Uprime = U - 2*Jhund)')
parser.add_argument('--beta', type=float, default=10.0, help='Inverse temperature')
parser.add_argument('--omega_0', type=float, default=1.0, help='Phonon/bosonic mode frequency')
parser.add_argument('--g', type=float, default=0.5, help='Electron-phonon coupling strength (coupling to total density is g**2 * Q(tau))')
parser.add_argument('--half_bandwidth', type=float, default=2.0, help='Half-bandwidth D of the Bethe-lattice (semicircular) bath, per orbital')
parser.add_argument('--n_cycles', type=int, default=1000000, help='Number of MC cycles')
parser.add_argument('--n_warmup_cycles', type=int, default=50000, help='Warmup cycles')
parser.add_argument('--length_cycle', type=int, default=100, help='Length of MC cycle')
parser.add_argument('--dyn_n_l', type=int, default=100, help='Number of Legendre polynomials for dynamical interactions')
parser.add_argument("--lang_firsov", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True, help="Whether to use Lang-Firsov for the dynamical part (False forces the stochastic double expansion)")
args, unknown = parser.parse_known_args()

n_orb = args.n_orb
U = args.U
J = args.Jhund
beta = args.beta
omega_0 = args.omega_0
g = args.g
half_bandwidth = args.half_bandwidth
# Half filling for h_int_kanamori's density-density matrix alone: each flavor couples
# to U (same orbital, opposite spin), U-2*J (different orbital, opposite spin), and
# U-3*J (different orbital, same spin), once each for n_orb=2; mu_half = (sum of those)/2.
# This ignores the dynamical (phonon) interaction's own static contribution -- it's
# corrected for below, after a cheap probe solve exposes how much extra it adds
# (see the "mu correction" block).
mu_bare = 0.5 * U + (n_orb - 1) * 0.5 * (U - 2 * J) + (n_orb - 1) * 0.5 * (U - 3 * J)

spin_names = ('up', 'down')
gf_struct = set_operator_structure(spin_names, n_orb, True)

# Retarded phonon kernel Q(tau), coupled uniformly to total density -- same closed form
# used throughout benchmark/dynamic_int and test/python/kanamori_dyn.py.
n_tau_bosonic = 2001
tau_mesh_pts = np.linspace(0, beta, n_tau_bosonic)
Q_data = -(1.0 / (2.0 * omega_0)) * np.cosh(omega_0 * (tau_mesh_pts - beta / 2.0)) / np.sinh(omega_0 * beta / 2.0)
Q_tau = GfImTime(indices=[0], beta=beta, statistic='Boson', n_points=n_tau_bosonic)
Q_tau.data[:, 0, 0] = g**2 * Q_data

# Hybridization: decoupled Bethe-lattice (semicircular) bath, same per orbital --
# Delta(iw) = (D/2)^2 * g_semicircular(iw), the standard fixed (non-self-consistent)
# Bethe-lattice hybridization, much smoother than a hard two-pole bath.
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

# Dynamical part: same coupling for U and Uprime (uniform over the whole group), no
# dynamical spin-flip channel.
kanamori_dynamical_vertices(S, spin_names, list(range(n_orb)), U=Q_tau, Uprime=Q_tau, spin_flip=False)
for s in spin_names:
    for a in range(n_orb):
        S.add_dyn_vertex(n(s, a), n(s, a), _as_scalar_gf(Q_tau))

solve_params = {
    "h_int": H_int,
    "length_cycle": args.length_cycle,
    "n_warmup_cycles": args.n_warmup_cycles,
    "n_cycles": args.n_cycles,
    "measure_pert_order": True,
    "lang_firsov": args.lang_firsov,
    "dyn_n_l": args.dyn_n_l,
}

# Mu correction: apply_lang_firsov_shift folds the phonon's static K'(0) response into
# h_loc as a genuine extra density-density interaction (standard Lang-Firsov/polaron
# physics) -- but mu_bare above only accounts for h_int's static U/J, not this extra
# interaction, so the model would sit away from half filling. A cheap probe solve is
# enough to expose the correction (apply_lang_firsov_shift runs once, deterministically,
# before any stochastic sampling): compare the *actual* renormalized mu the shift
# produced (S.lang_firsov_mu_renorm) against the half-filling mu implied by the *full*
# renormalized interaction matrix (S.lang_firsov_U_renorm, which already includes both
# h_int's own static coupling and the dynamical interaction's static part), then apply
# the same correction to mu_bare. No-op (mu_renorm empty) when lang_firsov=False, since
# the stochastic path samples the interaction directly and never touches h_loc this way.
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

# Save data
if mpi.is_master_node():
    filename = (f"/mnt/home/ahardy/ceph/CTHYB_Data/kanamori_dyn_cthyb_norb-{n_orb}_U-{U}_J-{J}"
                f"_g-{g}_w0-{omega_0}_beta-{beta}_nc-{args.n_cycles}_lf={args.lang_firsov}.h5")
    with h5.HDFArchive(filename, "w") as A:
        A['G_tau'] = S.G_tau
        A['perturbation_order'] = S.perturbation_order
        # perturbation_order_dyn is only measured (non-None) if at least one vertex was
        # routed to the stochastic path -- with the diagonal terms above, everything is
        # expected to recover to the analytic path instead (0 stochastic), so this is
        # routinely None here; a bare None can't be written to HDF5.
        if S.perturbation_order_dyn is not None:
            A['perturbation_order_dynamical'] = S.perturbation_order_dyn
        A['average_sign'] = S.average_sign
        A['K_n'] = S.K_n
        A['n_orb'] = n_orb
        A['U'] = U
        A['J'] = J
        A['g'] = g
        A['omega_0'] = omega_0
        A['beta'] = beta
        A['lang_firsov'] = args.lang_firsov
        A['mu_bare'] = mu_bare
        A['mu'] = mu
    print(f"Results saved to {filename}")
    print(f"Average sign = {S.average_sign}")
