# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""LxL cluster DMFT (DCA or CDMFT) for the single-band Hubbard model on the square lattice, 4x4 by default,

    H = -t sum_<ij>,s c^dag_is c_js + U sum_i n_i,up n_i,dn - mu sum_i,s n_i,s,

with the Pauli proposal of the pair moves mixed in at --pauli_prob (0.5 by default; neither scheme is in the regime
where it defaults to 1).

Both schemes work in a real cluster-momentum basis: K and -K combine into the waves cos(K.R) and sin(K.R) (one wave
when K = -K), gf_struct is 2 L^2 blocks of size 1, and h_int carries the momentum-conserving scattering of U, as in
plaquette_2x2.py. In DCA, G0, Delta and Sigma are diagonal in this basis by construction (G(K) = G(-K) by inversion).
In CDMFT they are diagonal only by the point group, which holds for 2x2 but not for 4x4, where it would take two blocks
of 16 orbitals: the script checks every iteration and stops otherwise. (In the site basis the 2x2 CDMFT sign is ~0.)
With eps(k) = -2t (cos kx + cos ky), k~ in the reduced zone [-pi/L, pi/L)^2 and the cluster momenta K = 2 pi (m, n) / L,

    DCA    G(K) = < [iw + mu - eps(K + k~) - Sigma(K)]^-1 >_k~
    CDMFT  G    = < [iw + mu - t(k~) - Sigma]^-1 >_k~,   t(k~)_RR' = 1/L^2 sum_K e^{i (K + k~).(R - R')} eps(K + k~)

and G0^-1 = G^-1 + Sigma. Sigma comes from the Legendre G by Dyson, its tail fitted as in plaquette_2x2.py.

Feasibility: CTHYB diagonalizes the cluster Hamiltonian in its invariant subspaces and keeps dense matrices of them.
Before building anything, the script bounds the largest subspace from below, as if the cluster momenta and the 8
point-group operations split the half-filled (N_up, N_dn) sector evenly, and compares one dense matrix of that size
with the memory of the node. For L = 4 that is about 1.3e6 states, 13 TB, and the script stops unless --force.
--probe stops after building the atomic problem.

    mpirun -n 8 python cluster_4x4.py --L 2 --scheme DCA --pauli_prob 0.5
"""
import argparse
import os
import sys
import time
from math import comb

import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.gfs import dyson
from triqs.operators import c, dagger
from triqs_cthyb import Solver
from triqs_cthyb.tail_fit import tail_fit

SPINS = ("up", "dn")

parser = argparse.ArgumentParser(description="LxL DCA / CDMFT for the square-lattice Hubbard model")
parser.add_argument("--L", type=int, default=4, help="Linear cluster size")
parser.add_argument("--scheme", choices=("DCA", "CDMFT"), default="DCA")
parser.add_argument("--U", type=float, default=6.0)
parser.add_argument("--t", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=10.0)
parser.add_argument("--mu", type=float, default=None, help="Chemical potential (default U/2, half filling)")
parser.add_argument("--pauli_prob", type=float, default=0.5, help="Probability of the Pauli proposal of the pair moves")
parser.add_argument("--n_loops", type=int, default=10)
parser.add_argument("--n_k", type=int, default=16, help="k~ points per direction of the reduced zone")
parser.add_argument("--n_l", type=int, default=40, help="Legendre coefficients for G_l")
parser.add_argument("--fit_min_w", type=float, default=4.0, help="Sigma is replaced by its fitted tail above this")
parser.add_argument("--fit_max_w", type=float, default=10.0)
parser.add_argument("--n_cycles", type=int, default=100000, help="QMC cycles per rank")
parser.add_argument("--n_warmup_cycles", type=int, default=5000)
parser.add_argument("--length_cycle", type=int, default=100)
parser.add_argument("--max_time", type=int, default=-1, help="Seconds per solve, -1 for none")
parser.add_argument("--probe", action="store_true", help="Only build the atomic problem and report its size")
parser.add_argument("--force", action="store_true", help="Go on even if the memory estimate exceeds the node")
parser.add_argument("--out_dir", default=".")
args = parser.parse_args()
mu = args.U / 2 if args.mu is None else args.mu
L, N = args.L, args.L**2

# ---- Feasibility, before any operator is built ----

largest = comb(N, N // 2) ** 2 / (N * 8)
need = 8.0 * largest**2
try:
    have = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
except (ValueError, OSError, AttributeError):
    have = float("nan")
mpi.report(f"{L}x{L} cluster, {2 * N} spin-orbitals, {2.0 ** (2 * N):.2e} states; half-filled (N_up, N_dn) sector "
           f"{comb(N, N // 2) ** 2:.2e} states, largest invariant subspace >= {largest:.2e} even with all cluster symmetries;"
           f" one dense matrix of it >= {need / 1e9:.3g} GB, node memory {have / 1e9:.3g} GB")
if need > have and not args.force:
    mpi.report("Stopping: the local problem cannot be diagonalized on this node (use --force to try anyway).")
    sys.exit(3)

# ---- Cluster, basis and interaction ----

SITES = np.array([[x, y] for y in range(L) for x in range(L)])  # R, and the (m, n) of K = 2 pi (m, n) / L
K_VECS = 2 * np.pi / L * SITES


def eps(k):
    return -2 * args.t * (np.cos(k[..., 0]) + np.cos(k[..., 1]))


def real_k_basis():
    """P[R, j] real orthogonal with columns cos(K.R) and sin(K.R) (one for K = -K), and the K index of each column"""
    cols, k_of, names, seen = [], [], [], set()
    for k, (m, n_) in enumerate(SITES):
        if k in seen:
            continue
        mk = ((-n_) % L) * L + (-m) % L
        phase = SITES @ K_VECS[k]
        if mk == k:
            cols, k_of, names = cols + [np.cos(phase) / L], k_of + [k], names + [f"K{m}{n_}"]
        else:
            cols += [np.sqrt(2) * np.cos(phase) / L, np.sqrt(2) * np.sin(phase) / L]
            k_of, names = k_of + [k, k], names + [f"K{m}{n_}c", f"K{m}{n_}s"]
        seen |= {k, mk}
    P = np.array(cols).T
    assert np.allclose(P.T @ P, np.eye(N)), "the cos/sin basis is not orthonormal"
    return P, np.array(k_of), names


P, K_OF, NAMES = real_k_basis()
GF_STRUCT = [(f"{s}_{name}", 1) for s in SPINS for name in NAMES]


def c_site(s, r):
    return sum(P[r, j] * c(f"{s}_{name}", 0) for j, name in enumerate(NAMES) if abs(P[r, j]) > 1e-14)


S = Solver(beta=args.beta, gf_struct=GF_STRUCT, n_l=args.n_l)
iw = np.array([complex(w) for w in S.G0_iw[GF_STRUCT[0][0]].mesh.values()])
iw0 = len(iw) // 2

k1 = (np.arange(args.n_k) + 0.5) * 2 * np.pi / (L * args.n_k) - np.pi / L
KT = np.stack(np.meshgrid(k1, k1, indexing="ij"), axis=-1).reshape(-1, 2)


def coarse_grain(sigma):
    """The cluster G of each wave j, shape (n_w, N), from sigma of each wave (n_w, N)"""
    zeta = iw + mu
    if args.scheme == "DCA":  # each wave with the dispersion of its K (the k~ grid is symmetric, so K and -K agree)
        return sum(1 / (zeta[:, None] - eps(kt + K_VECS[K_OF])[None, :] - sigma) for kt in KT) / len(KT)
    sigma_site = np.einsum("rj,wj,sj->wrs", P, sigma, P)
    G = 0
    for kt in KT:
        phase = np.exp(1j * SITES @ (K_VECS + kt).T) / L  # [R, K]
        h = (phase * eps(K_VECS + kt)) @ phase.conj().T
        G = G + np.linalg.inv(zeta[:, None, None] * np.eye(N) - h - sigma_site)
    G = P.T @ (G / len(KT)) @ P
    diag = np.diagonal(G, axis1=1, axis2=2)
    off = np.abs(G - diag[:, :, None] * np.eye(N)).max()
    if off > 1e-8 * np.abs(diag).max():
        mpi.report(f"Stopping: the CDMFT cluster G is not diagonal in the cos/sin K basis (max |G_jj'| = {off:.2e}); "
                   "this cluster needs blocks of L^2 orbitals.")
        sys.exit(4)
    return diag


def set_G0(G, sigma):
    G0 = 1 / (1 / G + sigma)
    for s in SPINS:
        for j, name in enumerate(NAMES):
            S.G0_iw[f"{s}_{name}"].data[:, 0, 0] = G0[:, j]


# Start from the Hartree self-energy of the paramagnetic half-filled solution
sigma = np.full((len(iw), N), args.U / 2, dtype=complex)
set_G0(coarse_grain(sigma), sigma)

# Built after the first coarse graining, which stops a CDMFT cluster that is not diagonal in this basis
h_int = args.U * sum(dagger(c_site("up", r)) * c_site("up", r) * dagger(c_site("dn", r)) * c_site("dn", r) for r in range(N))

params = dict(h_int=h_int, n_cycles=args.n_cycles, n_warmup_cycles=args.n_warmup_cycles, length_cycle=args.length_cycle,
              max_time=args.max_time, pauli_prob=args.pauli_prob, measure_G_l=True, measure_density_matrix=True,
              use_norm_as_weight=True)

if args.probe:
    t0 = time.time()
    S.solve(**{**params, "n_cycles": 0, "n_warmup_cycles": 0, "perform_post_proc": False})
    mpi.report(f"Atomic problem built in {time.time() - t0:.1f} s")
    sys.exit(0)

out = os.path.join(args.out_dir, f"cluster_{L}x{L}_{args.scheme}_U{args.U:.2f}_beta{args.beta:.1f}_mu{mu:.3f}_pauli{args.pauli_prob:.2f}.h5")
if mpi.is_master_node():
    with HDFArchive(out, "w") as A:
        A["params"] = {k: v for k, v in vars(args).items() if v is not None}
        A["mu"] = mu
        A["gf_struct"] = GF_STRUCT

for it in range(args.n_loops):
    t0 = time.time()
    S.solve(**params)
    wall = time.time() - t0

    G_iw = S.G0_iw.copy()
    for name, g in G_iw:
        g.set_from_legendre(S.G_l[name])
    Sigma_iw_raw = dyson(G0_iw=S.G0_iw, G_iw=G_iw)
    Sigma_iw = tail_fit(Sigma_iw_raw.copy(), fit_min_w=args.fit_min_w, fit_max_w=args.fit_max_w,
                        fit_max_moment=3, fit_known_moments=S.Sigma_moments)

    # Paramagnetic, and in DCA one Sigma per K, shared by its cos and sin waves
    sigma = np.mean([[Sigma_iw[f"{s}_{name}"].data[:, 0, 0] for name in NAMES] for s in SPINS], axis=0).T
    if args.scheme == "DCA":
        for k in set(K_OF):
            sigma[:, K_OF == k] = sigma[:, K_OF == k].mean(axis=1, keepdims=True)
    set_G0(coarse_grain(sigma), sigma)

    density = sum(occ[0, 0].real for occ in S.orbital_occupations.values()) / N
    sign = np.real(S.average_sign)
    sig0 = sigma[iw0]
    mpi.report(f"iteration {it}: <n> = {density:.4f}, sign = {sign:.4f}, order = {S.average_order:.1f}, "
               f"autocorrelation {S.auto_corr_time:.2f} cycles, solve {wall:.0f} s, max |Im Sigma(iw_0)| = {np.abs(sig0.imag).max():.4f}")

    if mpi.is_master_node():
        with HDFArchive(out, "a") as A:
            A[f"it{it}"] = dict(G0_iw=S.G0_iw, G_iw=G_iw, Sigma_iw=Sigma_iw, Sigma_iw_raw=Sigma_iw_raw, G_l=S.G_l,
                                density=density, average_sign=sign, average_order=S.average_order,
                                auto_corr_time=S.auto_corr_time, solve_seconds=wall)
            A["n_iter"] = it + 1
