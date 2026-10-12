# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""Cluster DMFT (DCA or CDMFT) for the single-band Hubbard model on the square lattice, solved by CTHYB: the clusters of
clusters.py (2x2, the 8-site Betts cluster 8A, 4x4; 4x4 by default),

    H = -t sum_<ij>,s c^dag_is c_js + U sum_i n_i,up n_i,dn - mu sum_i,s n_i,s,

with the Pauli proposal of the pair moves mixed in at --pauli_prob (0.5 by default; neither scheme is in the regime
where it defaults to 1).

Both schemes work in a real cluster-momentum basis: K and -K combine into the waves cos(K.R) and sin(K.R) (one wave
when K = -K), and h_int carries the momentum-conserving scattering of U, as in plaquette_2x2.py. The blocks of gf_struct
are the waves that the coarse-grained G of the Hartree start connects. In DCA, G0, Delta and Sigma are diagonal in this
basis by construction (G(K) = G(-K) by inversion), so every wave is a block of size 1. In CDMFT they are diagonal only
by the point group: the blocks are of size 1 for 2x2, larger for 8A, and every iteration checks that the coarse-grained G
has nothing outside them. (In the site basis the 2x2 CDMFT sign is ~0.) For 8A DCA the invariant subspaces CTHYB finds
in this basis have at most 660 states. With eps(k) = -2t (cos kx + cos ky), the cluster momenta K and k~ in the DCA
patch of clusters.py,

    DCA    G(K) = < [iw + mu - eps(K + k~) - Sigma(K)]^-1 >_k~
    CDMFT  G    = < [iw + mu - t(k~) - Sigma]^-1 >_k~,   t(k~)_RR' = 1/N sum_K e^{i (K + k~).(R - R')} eps(K + k~)

and G0^-1 = G^-1 + Sigma. Sigma comes from the Legendre G by Dyson, its tail fitted on [--fit_min_w, --fit_max_w] with
the known moments. At U = 6 the 1/w^3 tail only holds above w ~ 10: the [4, 10] of plaquette_2x2.py puts a kink at the
splice and is ~20% off up to w ~ 12, against CTINT.

Feasibility: CTHYB diagonalizes the cluster Hamiltonian in its invariant subspaces and keeps dense matrices of them.
Before building anything, the script bounds the largest subspace from below, as if the cluster momenta and the 8
point-group operations split the half-filled (N_up, N_dn) sector evenly, and compares one dense matrix of that size
with the memory of the node. For 4x4 that is about 1.3e6 states, 13 TB, and the script stops unless --force.
--probe stops after building the atomic problem.

    mpirun -n 8 python cluster_4x4.py --cluster 8A --scheme DCA --pauli_prob 0.5
"""
import argparse
import os
import sys
import time
from math import comb

import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.gfs import MeshImFreq, dyson
from triqs.operators import c, dagger
from triqs_cthyb import Solver
from triqs_cthyb.tail_fit import tail_fit

from clusters import SUPERLATTICES, Cluster

SPINS = ("up", "dn")
N_IW = 1025

parser = argparse.ArgumentParser(description="Cluster DCA / CDMFT for the square-lattice Hubbard model, solved by CTHYB")
parser.add_argument("--cluster", choices=tuple(SUPERLATTICES), default="4x4")
parser.add_argument("--scheme", choices=("DCA", "CDMFT"), default="DCA")
parser.add_argument("--U", type=float, default=6.0)
parser.add_argument("--t", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=10.0)
parser.add_argument("--mu", type=float, default=None, help="Chemical potential (default U/2, half filling)")
parser.add_argument("--pauli_prob", type=float, default=0.5, help="Probability of the Pauli proposal of the pair moves")
parser.add_argument("--n_loops", type=int, default=10)
parser.add_argument("--n_k", type=int, default=16, help="k~ points per direction of the reduced zone")
parser.add_argument("--n_l", type=int, default=40, help="Legendre coefficients for G_l")
parser.add_argument("--fit_min_w", type=float, default=10.0, help="Sigma is replaced by its fitted tail above this")
parser.add_argument("--fit_max_w", type=float, default=20.0)
parser.add_argument("--n_cycles", type=int, default=100000, help="QMC cycles per rank")
parser.add_argument("--n_warmup_cycles", type=int, default=5000)
parser.add_argument("--length_cycle", type=int, default=100)
parser.add_argument("--max_time", type=int, default=-1, help="Seconds per solve, -1 for none")
parser.add_argument("--probe", action="store_true", help="Only build the atomic problem and report its size")
parser.add_argument("--force", action="store_true", help="Go on even if the memory estimate exceeds the node")
parser.add_argument("--out_dir", default=".")
args = parser.parse_args()
mu = args.U / 2 if args.mu is None else args.mu
N = int(round(abs(np.linalg.det(np.array(SUPERLATTICES[args.cluster], dtype=float)))))

# ---- Feasibility, before any operator is built ----

largest = comb(N, N // 2) ** 2 / (N * 8)
need = 8.0 * largest**2
try:
    have = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
except (ValueError, OSError, AttributeError):
    have = float("nan")
mpi.report(f"{args.cluster} cluster, {2 * N} spin-orbitals, {2.0 ** (2 * N):.2e} states; half-filled (N_up, N_dn) sector "
           f"{comb(N, N // 2) ** 2:.2e} states, largest invariant subspace >= {largest:.2e} even with all cluster symmetries;"
           f" one dense matrix of it >= {need / 1e9:.3g} GB, node memory {have / 1e9:.3g} GB")
if need > have and not args.force:
    mpi.report("Stopping: the local problem cannot be diagonalized on this node (use --force to try anyway).")
    sys.exit(3)

# ---- Cluster and lattice ----

CL = Cluster(args.cluster, args.n_k)
P, K_OF, NAMES = CL.P, CL.K_of, CL.wave_names


def eps(k):
    return -2 * args.t * (np.cos(k[..., 0]) + np.cos(k[..., 1]))


iw = np.array([complex(w) for w in MeshImFreq(beta=args.beta, S="Fermion", n_iw=N_IW).values()])
iw0 = len(iw) // 2

KT = CL.k_patch


def coarse_grain(sigma):
    """The cluster G in the wave basis, (n_w, N, N), from Sigma in the wave basis"""
    zeta = iw + mu
    if args.scheme == "DCA":  # each wave with the dispersion of its K (the k~ grid is symmetric, so K and -K agree)
        s = np.diagonal(sigma, axis1=1, axis2=2)
        return (sum(1 / (zeta[:, None] - eps(kt + CL.K[K_OF])[None, :] - s) for kt in KT) / len(KT))[:, :, None] * np.eye(N)
    sigma_site = P @ sigma @ P.T
    G = 0
    for kt in KT:
        G = G + np.linalg.inv(zeta[:, None, None] * np.eye(N) - CL.superlattice_hamiltonian(eps, kt) - sigma_site)
    return P.T @ (G / len(KT)) @ P


# Start from the Hartree self-energy of the paramagnetic half-filled solution
sigma = np.full(len(iw), args.U / 2, dtype=complex)[:, None, None] * np.eye(N)
G = coarse_grain(sigma)

# ---- Blocks: the waves connected by the coarse-grained G ----

linked = np.abs(G).max(axis=0) > 1e-8 * np.abs(G).max()
BLOCKS = []
for j in range(N):
    if any(j in b for b in BLOCKS):
        continue
    block = {j}
    while (grown := block | set(np.nonzero(linked[sorted(block)].any(axis=0))[0])) != block:
        block = grown
    BLOCKS.append(sorted(block))
BLOCK_NAMES = ["|".join(NAMES[j] for j in b) for b in BLOCKS]
GF_STRUCT = [(f"{s}_{name}", len(b)) for s in SPINS for name, b in zip(BLOCK_NAMES, BLOCKS)]
WHERE = {j: (name, i) for name, b in zip(BLOCK_NAMES, BLOCKS) for i, j in enumerate(b)}  # wave -> (block, index)
mpi.report(f"{args.scheme} blocks of the wave basis: {', '.join(BLOCK_NAMES)}")


def block_view(m, b):
    return m[np.ix_(np.arange(len(iw)), b, b)]


def check_blocks(G):
    """Stop if the coarse-grained G has weight outside the blocks"""
    inside = np.zeros((N, N), dtype=bool)
    for b in BLOCKS:
        inside[np.ix_(b, b)] = True
    off = np.abs(G[:, ~inside]).max(initial=0.0)
    if off > 1e-8 * np.abs(G).max():
        mpi.report(f"Stopping: the coarse-grained G leaves the blocks (max |G_jj'| outside = {off:.2e}).")
        sys.exit(4)


def c_site(s, r):
    return sum(P[r, j] * c(f"{s}_{WHERE[j][0]}", WHERE[j][1]) for j in range(N) if abs(P[r, j]) > 1e-14)


S = Solver(beta=args.beta, gf_struct=GF_STRUCT, n_iw=N_IW, n_l=args.n_l)


def set_G0(G, sigma):
    for name, b in zip(BLOCK_NAMES, BLOCKS):
        G0 = np.linalg.inv(np.linalg.inv(block_view(G, b)) + block_view(sigma, b))
        for s in SPINS:
            S.G0_iw[f"{s}_{name}"].data[:] = G0


set_G0(G, sigma)

h_int = args.U * sum(dagger(c_site("up", r)) * c_site("up", r) * dagger(c_site("dn", r)) * c_site("dn", r) for r in range(N))

params = dict(h_int=h_int, n_cycles=args.n_cycles, n_warmup_cycles=args.n_warmup_cycles, length_cycle=args.length_cycle,
              max_time=args.max_time, pauli_prob=args.pauli_prob, measure_G_l=True, measure_density_matrix=True,
              use_norm_as_weight=True, imag_threshold=1e-10)  # the k~ sum leaves ~1e-13 in Im Delta(infty) of CDMFT blocks

if args.probe:
    t0 = time.time()
    S.solve(**{**params, "n_cycles": 0, "n_warmup_cycles": 0, "perform_post_proc": False})
    mpi.report(f"Atomic problem built in {time.time() - t0:.1f} s")
    sys.exit(0)

out = os.path.join(args.out_dir, f"cluster_{args.cluster}_{args.scheme}_U{args.U:.2f}_beta{args.beta:.1f}_mu{mu:.3f}_pauli{args.pauli_prob:.2f}.h5")
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
    sigma = np.zeros((len(iw), N, N), dtype=complex)
    for name, b in zip(BLOCK_NAMES, BLOCKS):
        sigma[np.ix_(np.arange(len(iw)), b, b)] = np.mean([Sigma_iw[f"{s}_{name}"].data for s in SPINS], axis=0)
    if args.scheme == "DCA":
        for k in set(K_OF):
            js = np.nonzero(K_OF == k)[0]
            sigma[:, js, js] = sigma[:, js, js].mean(axis=1, keepdims=True)
    G = coarse_grain(sigma)
    check_blocks(G)
    set_G0(G, sigma)

    density = sum(np.trace(occ).real for occ in S.orbital_occupations.values()) / N
    sign = np.real(S.average_sign)
    sig0 = np.diagonal(sigma[iw0])
    mpi.report(f"iteration {it}: <n> = {density:.4f}, sign = {sign:.4f}, order = {S.average_order:.1f}, "
               f"autocorrelation {S.auto_corr_time:.2f} cycles, solve {wall:.0f} s, max |Im Sigma(iw_0)| = {np.abs(sig0.imag).max():.4f}")

    if mpi.is_master_node():
        with HDFArchive(out, "a") as A:
            A[f"it{it}"] = dict(G0_iw=S.G0_iw, G_iw=G_iw, Sigma_iw=Sigma_iw, Sigma_iw_raw=Sigma_iw_raw, G_l=S.G_l,
                                density=density, average_sign=sign, average_order=S.average_order,
                                auto_corr_time=S.auto_corr_time, solve_seconds=wall)
            A["n_iter"] = it + 1
