# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""CTINT counterpart of cluster_4x4.py: the same cluster DCA / CDMFT (clusters.py) for the square-lattice Hubbard model,

    H = -t sum_<ij>,s c^dag_is c_js + U sum_i n_i,up n_i,dn - mu sum_i,s n_i,s,

with the same coarse graining, starting point and per-iteration report, to compare the solvers.

CTINT expands in density-density vertices (n_a - alpha)(n_b - alpha), so it works in the site basis, where U is one
such term per site: gf_struct is one block of N sites per spin, in DCA as in CDMFT. In DCA the coarse graining is
done per cluster momentum in the real cos/sin basis of cluster_4x4.py and rotated back to the sites. The alpha shifts
are the signed ones of ../dynamic_int/common/ctint.py, and CTINT takes G0 on its DLR mesh: G0 is built on a regular
Matsubara mesh, Fourier transformed and DLR-fitted, as in the dynamic_int CTINT runs. Sigma = G0^-1 - G^-1 on the
regular mesh, from CTINT's G (a DLR expansion, so its high-frequency tail is smooth).

    mpirun -n 8 python cluster_ctint.py --cluster 4x4 --scheme DCA
"""
import argparse
import os
import sys
import time

import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.gfs import Gf, MeshImFreq, fit_gf_dlr, make_gf_dlr_imfreq, make_gf_from_fourier
from triqs.operators import n
from triqs_ctint import Solver

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dynamic_int"))
from common import ctint  # noqa: E402  (alpha shifts and DLR conversions of the dynamic_int CTINT runs)
from clusters import SUPERLATTICES, Cluster  # noqa: E402

SPINS = ("up", "dn")

parser = argparse.ArgumentParser(description="Cluster DCA / CDMFT for the square-lattice Hubbard model, solved by CTINT")
parser.add_argument("--cluster", choices=tuple(SUPERLATTICES), default="4x4")
parser.add_argument("--scheme", choices=("DCA", "CDMFT"), default="DCA")
parser.add_argument("--U", type=float, default=6.0)
parser.add_argument("--t", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=10.0)
parser.add_argument("--mu", type=float, default=None, help="Chemical potential (default U/2, half filling)")
parser.add_argument("--n_loops", type=int, default=10)
parser.add_argument("--n_k", type=int, default=16, help="k~ points per direction of the reduced zone")
parser.add_argument("--n_iw", type=int, default=1025, help="Matsubara frequencies of the self-consistency")
parser.add_argument("--n_tau", type=int, default=10001, help="tau points of the Fourier transform before the DLR fit")
parser.add_argument("--dlr_wmax", type=float, default=20.0, help="DLR frequency cutoff")
parser.add_argument("--dlr_eps", type=float, default=1e-10, help="DLR accuracy of the G0 fit; must give CTINT's own DLR mesh")
parser.add_argument("--n_cycles", type=int, default=100000, help="QMC cycles per rank")
parser.add_argument("--n_warmup_cycles", type=int, default=5000)
parser.add_argument("--length_cycle", type=int, default=100)
parser.add_argument("--max_time", type=int, default=-1, help="Seconds per solve, -1 for none")
parser.add_argument("--out_dir", default=".")
ctint.add_alpha_args(parser)
args = parser.parse_args()
mu = args.U / 2 if args.mu is None else args.mu

# ---- Cluster and lattice, as in cluster_4x4.py ----

CL = Cluster(args.cluster, args.n_k)
P, K_OF, N = CL.P, CL.K_of, CL.N


def eps(k):
    return -2 * args.t * (np.cos(k[..., 0]) + np.cos(k[..., 1]))


GF_STRUCT = [(s, N) for s in SPINS]
mesh = MeshImFreq(beta=args.beta, S="Fermion", n_iw=args.n_iw)
iw = np.array([complex(w) for w in mesh.values()])
iw0 = len(iw) // 2

KT = CL.k_patch


def coarse_grain(sigma):
    """The cluster G in the site basis, (n_w, N, N), from Sigma in the site basis"""
    zeta = iw + mu
    if args.scheme == "DCA":  # per wave of the cos/sin basis, with the dispersion of its K
        sigma_w = np.einsum("rj,wrs,sj->wj", P, sigma, P)
        for k in set(K_OF):  # the cos and sin waves of a K share Sigma(K)
            sigma_w[:, K_OF == k] = sigma_w[:, K_OF == k].mean(axis=1, keepdims=True)
        G_w = sum(1 / (zeta[:, None] - eps(kt + CL.K[K_OF])[None, :] - sigma_w) for kt in KT) / len(KT)
        return np.einsum("rj,wj,sj->wrs", P, G_w, P)
    G = 0
    for kt in KT:
        G = G + np.linalg.inv(zeta[:, None, None] * np.eye(N) - CL.superlattice_hamiltonian(eps, kt) - sigma)
    return G / len(KT)


def solver_for(G0):
    """A new CTINT solver whose DLR G0_iw, in both spin blocks, is G0 (n_w, N, N) of the regular mesh. One solver per
    iteration: the runs of 2026-10-09 that reused one all stopped at the start of the second solve."""
    S = Solver(beta=args.beta, gf_struct=GF_STRUCT, n_tau=args.n_tau, use_D=False, dlr_wmax=args.dlr_wmax)
    g0_iw = Gf(mesh=mesh, target_shape=(N, N))
    g0_iw.data[:] = G0
    g0_dlr = make_gf_dlr_imfreq(fit_gf_dlr(make_gf_from_fourier(g0_iw, args.n_tau), w_max=args.dlr_wmax, eps=args.dlr_eps,
                                           symmetrize=True))
    for s in SPINS:
        S.G0_iw[s].data[:] = g0_dlr.data[:]
    return S


# Start from the Hartree self-energy of the paramagnetic half-filled solution
sigma = args.U / 2 * np.eye(N) * np.ones((len(iw), 1, 1), dtype=complex)
G0 = np.linalg.inv(np.linalg.inv(coarse_grain(sigma)) + sigma)

h_int = args.U * sum(n("up", r) * n("dn", r) for r in range(N))
alpha_kwargs, alpha_report = ctint.alpha_kwargs(args, h_int, [])
mpi.report(alpha_report)

out = os.path.join(args.out_dir, f"cluster_{args.cluster}_{args.scheme}_U{args.U:.2f}_beta{args.beta:.1f}_mu{mu:.3f}_ctint.h5")
if mpi.is_master_node():
    with HDFArchive(out, "w") as A:
        A["params"] = {k: v for k, v in vars(args).items() if v is not None}
        A["mu"] = mu
        A["gf_struct"] = GF_STRUCT

for it in range(args.n_loops):
    S = solver_for(G0)
    t0 = time.time()
    S.solve(h_int=h_int, **alpha_kwargs, length_cycle=args.length_cycle, n_warmup_cycles=args.n_warmup_cycles,
            n_cycles=args.n_cycles, max_time=args.max_time, measure_M_iw=True, measure_M_tau=False, post_process=True)
    wall = time.time() - t0

    # Paramagnetic Sigma by Dyson on the regular mesh
    G = {s: ctint.to_regular_imfreq(S.G_iw[s], mesh).data for s in SPINS}
    sigma = np.mean([np.linalg.inv(G0) - np.linalg.inv(G[s]) for s in SPINS], axis=0)
    G0 = np.linalg.inv(np.linalg.inv(coarse_grain(sigma)) + sigma)

    density = sum(-np.trace(ctint.to_uniform_tau(S.G_iw[s], args.n_tau).data[-1]).real for s in SPINS) / N
    sign = float(np.real(S.average_sign))
    sig0 = np.diagonal(sigma[iw0])
    mpi.report(f"iteration {it}: <n> = {density:.4f}, sign = {sign:.4f}, order = {S.average_k:.1f}, solve {wall:.0f} s, "
               f"max |Im Sigma_ii(iw_0)| = {np.abs(sig0.imag).max():.4f}")

    if mpi.is_master_node():
        with HDFArchive(out, "a") as A:
            A[f"it{it}"] = dict(Sigma=sigma, G0=G0, G_iw=S.G_iw, density=density, average_sign=sign,
                                average_order=float(np.real(S.average_k)), solve_seconds=wall)
            A["n_iter"] = it + 1
