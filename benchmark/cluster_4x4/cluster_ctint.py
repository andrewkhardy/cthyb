# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""CTINT counterpart of cluster_4x4.py: the same DCA / CDMFT for the square-lattice Hubbard model, on an LxL cluster
or, in DCA, the Betts cluster 8A (cluster_geometry.py),

    H = -t sum_<ij>,s c^dag_is c_js + U sum_i n_i,up n_i,dn - mu sum_i,s n_i,s,

with the same coarse graining, starting point and per-iteration report, to compare the solvers.

CTINT expands in density-density vertices (n_a - alpha)(n_b - alpha), so it works in the site basis, where U is one
such term per site: gf_struct is one block of N sites per spin, in DCA as in CDMFT. In DCA the coarse graining is
done per cluster momentum in the real cos/sin basis of cluster_geometry.py and rotated back to the sites. The alpha
shifts are the signed ones of ../dynamic_int/common/ctint.py, and CTINT takes G0 on its DLR mesh: G0 is built on a
regular Matsubara mesh, Fourier transformed and DLR-fitted, as in the dynamic_int CTINT runs.

Sigma comes from CTINT's M(tau), binned on n_tau points, rather than from its G:
  - M is made real and symmetric, as it is here (real hopping, no field), and DLR-fitted (fit_M) by least squares
    weighted by its noise, at the largest eps that resolves it, so that no DLR coefficient fits noise, and with its
    1/iw moment fixed to the exact M_1 = Sigma_1 + M_hartree^2, Sigma_1 = U^2 n_i (1 - n_i) on the diagonal, with
    the density n_i per spin averaged over the spins and, in DCA, over the sites;
  - Sigma = Sigma_alpha + (1 + M G0_shift)^-1 M, with M = M_hartree + M_dyn(iw) and G0_shift^-1 = G0^-1 - Sigma_alpha.
    M_dyn decays as 1/iw, so Sigma tends to Sigma_hartree + Sigma_1 / iw with no tail fit, whereas Dyson from G,
    G0^-1 - G^-1, grows as iw times the error of the 1/iw moment of G, and is not symmetric to the 1e-12 that CTINT
    asks of G0.
In DCA, Sigma is then reduced to its DCA form, diagonal in the waves and shared by the cos and sin waves of a K. The
report gives the eps of each spin and Sigma_1 as the data alone give it, against the exact value imposed.

    mpirun -n 8 python cluster_ctint.py --cluster 4x4 --scheme DCA
"""
import argparse
import os
import sys
import time

import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.gfs import (Gf, MeshDLR, MeshImFreq, fit_gf_dlr, make_gf_dlr_imfreq, make_gf_from_fourier, make_gf_imfreq,
                        make_gf_imtime)
from triqs.operators import n
from triqs_ctint import Solver

from cluster_geometry import BETTS, geometry

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dynamic_int"))
from common import ctint  # noqa: E402  (the alpha shifts of the dynamic_int CTINT runs)

SPINS = ("up", "dn")
EPS_SCAN = [10.0**-k for k in range(2, 11)]  # DLR eps tried for M, coarse to fine
EPS_NOISE = 1e-7  # beyond it, a finer basis can only gain chi^2 on the noise of QMC data
NOISE_WINDOW = 101  # tau bins over which the noise of M(tau) is averaged

parser = argparse.ArgumentParser(description="Cluster DCA / CDMFT for the square-lattice Hubbard model, solved by CTINT")
parser.add_argument("--cluster", default="4x4", help="LxL, or the Betts cluster 8A (DCA only)")
parser.add_argument("--scheme", choices=("DCA", "CDMFT"), default="DCA")
parser.add_argument("--U", type=float, default=6.0)
parser.add_argument("--t", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=10.0)
parser.add_argument("--mu", type=float, default=None, help="Chemical potential (default U/2, half filling)")
parser.add_argument("--n_loops", type=int, default=10)
parser.add_argument("--n_k", type=int, default=16, help="k~ points per direction of the reduced zone")
parser.add_argument("--n_iw", type=int, default=1025, help="Matsubara frequencies of the self-consistency")
parser.add_argument("--n_tau", type=int, default=10001, help="tau points of the G0 Fourier transform and of the M(tau) bins")
parser.add_argument("--dlr_wmax", type=float, default=20.0, help="DLR frequency cutoff")
parser.add_argument("--dlr_eps", type=float, default=1e-10, help="DLR accuracy of CTINT's mesh and of the G0 fit")
parser.add_argument("--n_cycles", type=int, default=100000, help="QMC cycles per rank")
parser.add_argument("--n_warmup_cycles", type=int, default=5000)
parser.add_argument("--length_cycle", type=int, default=100)
parser.add_argument("--max_time", type=int, default=-1, help="Seconds per solve, -1 for none")
parser.add_argument("--out_dir", default=".")
ctint.add_alpha_args(parser)
args = parser.parse_args()
if args.scheme == "CDMFT" and args.cluster in BETTS:
    parser.error("CDMFT takes an LxL cluster: a Betts superlattice does not fix the shape of the CDMFT cluster")
mu = args.U / 2 if args.mu is None else args.mu
N, SITES, K_VECS, P, K_OF, _, KT = geometry(args.cluster, args.n_k)


def eps(k):
    return -2 * args.t * (np.cos(k[..., 0]) + np.cos(k[..., 1]))


GF_STRUCT = [(s, N) for s in SPINS]
mesh = MeshImFreq(beta=args.beta, S="Fermion", n_iw=args.n_iw)
iw = np.array([complex(w) for w in mesh.values()])
iw0 = len(iw) // 2


def dca_sigma(sigma):
    """Sigma in the site basis reduced to its DCA form: diagonal in the cos/sin waves, one value per K"""
    sigma_w = np.einsum("rj,wrs,sj->wj", P, sigma, P)
    for k in set(K_OF):  # the cos and sin waves of a K share Sigma(K)
        sigma_w[:, K_OF == k] = sigma_w[:, K_OF == k].mean(axis=1, keepdims=True)
    return np.einsum("rj,wj,sj->wrs", P, sigma_w, P)


def coarse_grain(sigma):
    """The cluster G in the site basis, (n_w, N, N), from Sigma in the site basis (in DCA, in its DCA form)"""
    zeta = iw + mu
    if args.scheme == "DCA":  # per wave of the cos/sin basis, with the dispersion of its K
        sigma_w = np.einsum("rj,wrs,sj->wj", P, sigma, P)
        G_w = sum(1 / (zeta[:, None] - eps(kt + K_VECS[K_OF])[None, :] - sigma_w) for kt in KT) / len(KT)
        return np.einsum("rj,wj,sj->wrs", P, G_w, P)
    G = 0
    for kt in KT:
        phase = np.exp(1j * SITES @ (K_VECS + kt).T) / np.sqrt(N)  # [R, K]
        h = (phase * eps(K_VECS + kt)) @ phase.conj().T
        G = G + np.linalg.inv(zeta[:, None, None] * np.eye(N) - h - sigma)
    return G / len(KT)


S = Solver(beta=args.beta, gf_struct=GF_STRUCT, n_tau=args.n_tau, use_D=False, dlr_wmax=args.dlr_wmax,
           dlr_eps=args.dlr_eps)


def set_G0(G0):
    """Hand G0 (n_w, N, N), on the regular mesh, to both spin blocks of CTINT's DLR G0_iw"""
    g0_iw = Gf(mesh=mesh, target_shape=(N, N))
    g0_iw.data[:] = G0
    g0_dlr = make_gf_dlr_imfreq(fit_gf_dlr(make_gf_from_fourier(g0_iw, args.n_tau), w_max=args.dlr_wmax, eps=args.dlr_eps,
                                           symmetrize=True))
    # CTINT checks G0_ij(iw) = G0_ji(-iw)* to 1e-12 on its node pairs (k, n - 1 - k), which the rounding of the fit
    # (~1e-12 at eps = 1e-10) can miss; the average of each pair satisfies it exactly
    g0_dlr.data[:] = (g0_dlr.data + g0_dlr.data[::-1].conj().transpose(0, 2, 1)) / 2
    for s in SPINS:
        S.G0_iw[s].data[:] = g0_dlr.data[:]


def symmetric(m):
    """m (..., N, N) made real and symmetric, as M and Sigma are here"""
    return ((m + np.swapaxes(m, -1, -2)) / 2).real


def dlr_basis(e):
    """The DLR mesh at eps e, and its basis functions on the n_tau points of M(tau) as columns"""
    dlr = MeshDLR(beta=args.beta, statistic="Fermion", w_max=args.dlr_wmax, eps=e, symmetrize=True)
    g = Gf(mesh=dlr, target_shape=[])
    cols = []
    for l in range(len(dlr)):
        g.data[:] = 0
        g.data[l] = 1
        cols.append(make_gf_imtime(g, args.n_tau).data.real.copy())
    return dlr, np.array(cols).T


BASES = [dlr_basis(e) for e in EPS_SCAN]


def lsq(K, y, var, a=None, b=0.0):
    """argmin_c sum_t (K c - y)_t^2 / var_t, under a.c = b if a is given"""
    Kw, yw = K / np.sqrt(var)[:, None], y / np.sqrt(var)
    c = np.linalg.lstsq(Kw, yw, rcond=None)[0]
    if a is not None:
        h = np.linalg.solve(Kw.T @ Kw, a)
        c -= h * (a @ c - b) / (a @ h)
    return c


def fit_M(m_tau, M1):
    """M_dyn(iw) (n_w, N, N) on the regular mesh, from the binned M_dyn(tau) (n_tau, N, N), real and symmetric, with its
    1/iw moment fixed to M1; the eps of the fit; and the 1/iw moment that the data alone give.

    Each element is fitted by least squares weighted by the noise of its bins, the residual of the finest fit averaged
    over NOISE_WINDOW bins (it is largest near tau = 0, beta). eps is the largest of EPS_SCAN whose chi^2 exceeds the
    finest fit's by no more than pure noise would over the extra coefficients (within 3 sigma): a finer basis spends
    them on noise, mostly near tau = 0, beta, which is where the 1/iw moment and the tail come from. The noise
    estimate is scaled by what the coefficients finer than EPS_NOISE gain per coefficient, which is noise alone."""
    i, j = np.triu_indices(N)
    y = m_tau[:, i, j]
    K = BASES[-1][1]
    resid2 = (y - K @ np.linalg.lstsq(K, y, rcond=None)[0]) ** 2
    pad, box = NOISE_WINDOW // 2, np.ones(NOISE_WINDOW) / NOISE_WINDOW
    var = np.array([np.convolve(np.pad(r, pad, mode="edge"), box, mode="valid") for r in resid2.T]).T
    var[[0, -1]] *= 2  # the first and last bins are half bins

    def fit(K, a=None):
        c = np.array([lsq(K, y[:, e], var[:, e], a, M1[i[e], j[e]]) for e in range(len(i))]).T
        return c, np.sum((y - K @ c) ** 2 / var)

    fits = [fit(K) for _, K in BASES]
    gain = [chi2 - fits[-1][1] for _, chi2 in fits]
    extra = [len(i) * (len(BASES[-1][0]) - len(dlr)) for dlr, _ in BASES]
    ref = EPS_SCAN.index(EPS_NOISE)
    phi = max(1.0, gain[ref] / extra[ref])
    pick = next(k for k in range(len(BASES)) if gain[k] / phi <= extra[k] + 3 * np.sqrt(2 * extra[k]))
    dlr, K = BASES[pick]
    a = -(K[0] + K[-1])  # the 1/iw moment of each basis function, -(f(0) + f(beta))
    moment_free = np.zeros((N, N))
    moment_free[i, j] = moment_free[j, i] = a @ fits[pick][0]
    g = Gf(mesh=dlr, target_shape=(N, N))
    g.data[:, i, j] = g.data[:, j, i] = fit(K, a)[0]
    return make_gf_imfreq(g, args.n_iw).data, EPS_SCAN[pick], moment_free


def sigma_from_M(b, G0, n_site):
    """Sigma (n_w, N, N) of spin block b on the regular mesh, from CTINT's M, the G0 of the solve and the density per
    site and spin; the eps of the fit of M; and the diagonal Sigma_1 that the data alone give and its exact value,
    averaged over the sites"""
    sigma_1 = args.U**2 * n_site * (1 - n_site)
    M_hartree = symmetric(S.M_hartree[b])
    M_dyn, eps_M, M1_free = fit_M(symmetric(S.M_tau[SPINS[b]].data), np.diag(sigma_1) + M_hartree @ M_hartree)
    sigma_alpha = symmetric(S.Sigma_hartree[b]) - M_hartree
    M = M_dyn + M_hartree
    G0_shift = np.linalg.inv(np.linalg.inv(G0) - sigma_alpha)
    sigma_1_free = np.diagonal(M1_free - M_hartree @ M_hartree).mean()
    return sigma_alpha + np.linalg.solve(np.eye(N) + M @ G0_shift, M), eps_M, (sigma_1_free, sigma_1.mean())


# Start from the Hartree self-energy of the paramagnetic half-filled solution
sigma = args.U / 2 * np.eye(N) * np.ones((len(iw), 1, 1), dtype=complex)
G0 = np.linalg.inv(np.linalg.inv(coarse_grain(sigma)) + sigma)
set_G0(G0)

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
    t0 = time.time()
    S.solve(h_int=h_int, **alpha_kwargs, length_cycle=args.length_cycle, n_warmup_cycles=args.n_warmup_cycles,
            n_cycles=args.n_cycles, max_time=args.max_time, measure_M_tau=True, post_process=True)
    wall = time.time() - t0

    # Paramagnetic Sigma from M, in DCA in its DCA form
    n_site = np.mean([np.diagonal(d).real for d in S.density], axis=0)
    if args.scheme == "DCA":  # whose cluster is translation invariant
        n_site[:] = n_site.mean()
    sigmas, eps_M, sigma_1 = zip(*(sigma_from_M(b, G0, n_site) for b in range(len(SPINS))))
    sigma = np.mean(sigmas, axis=0)
    if args.scheme == "DCA":
        sigma = dca_sigma(sigma)
    G0 = np.linalg.inv(np.linalg.inv(coarse_grain(sigma)) + sigma)
    set_G0(G0)

    density = 2 * n_site.mean()
    sign = float(np.real(S.average_sign))
    sig0 = np.diagonal(sigma[iw0])
    mpi.report(f"iteration {it}: <n> = {density:.4f}, sign = {sign:.4f}, order = {S.average_k:.1f}, solve {wall:.0f} s, "
               f"max |Im Sigma_ii(iw_0)| = {np.abs(sig0.imag).max():.4f}; M fitted at eps "
               f"{', '.join(f'{e:.0e}' for e in eps_M)}, Sigma_1 from the data "
               f"{', '.join(f'{d:.3f}' for d, _ in sigma_1)} against {', '.join(f'{x:.3f}' for _, x in sigma_1)} imposed")

    if mpi.is_master_node():
        with HDFArchive(out, "a") as A:
            A[f"it{it}"] = dict(Sigma=sigma, G0=G0, G_iw=S.G_iw, density=density, average_sign=sign,
                                average_order=float(np.real(S.average_k)), solve_seconds=wall, M_eps=list(eps_M),
                                sigma_1=np.array(sigma_1))
            A["n_iter"] = it + 1
