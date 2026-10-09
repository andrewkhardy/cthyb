# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""2x2 cluster DMFT (DCA or CDMFT) for the single-band Hubbard model on the square lattice,

    H = -t sum_<ij>,s c^dag_is c_js + U sum_i n_i,up n_i,dn - mu sum_i,s n_i,s.

The impurity is solved in the cluster-momentum basis

    c_K,s = 1/2 sum_R e^{-i K.R} c_R,s,    K = G (0,0), X (pi,0), Y (0,pi), M (pi,pi),

in which the Weiss field is diagonal. The four K states are the joint eigenvectors of the two
mirror planes of the plaquette (equivalently, of the 2x2 cluster translations), with four
distinct pairs of eigenvalues, so a one-body quantity with those symmetries has no K != K'
elements: G0, Delta and Sigma, in DCA by construction and in CDMFT by the point group of the
superlattice. gf_struct is therefore 8 blocks of size 1. The cost moves into h_int, where the
Hubbard term becomes the momentum-conserving

    U/4 sum_{K,K',Q} c^dag_{K+Q,up} c_{K,up} c^dag_{K'-Q,dn} c_{K',dn},

built below by rotating the site operators. Autopartition finds N_up, N_dn and the total K.

Self-consistency, with eps(k) = -2t (cos kx + cos ky) and k~ in the reduced zone [-pi/2, pi/2)^2:

    DCA    G(K) = < [iw + mu - eps(K + k~) - Sigma(K)]^-1 >_k~
    CDMFT  G    = < [iw + mu - t(k~) - Sigma]^-1 >_k~,   t(k~) = V(k~) diag(eps(K + k~)) V(k~)^dag

where V(k~) = P^T diag(e^{i k~.R}) P mixes the K states, so the CDMFT average is diagonal only
by symmetry; that is checked every iteration. Then G0(K)^-1 = G(K)^-1 + Sigma(K). The average
sign is markedly lower in CDMFT than in DCA (~0.8 against ~0.99 at the defaults).

Sigma comes from the Legendre G by Dyson, with its tail above fit_min_w replaced by a fit that
holds the density-matrix moments Sigma_0 (Hartree) and Sigma_1 fixed. The raw Dyson tail is
not usable here: G_l pins the 1/iw moment of G but not the 1/iw^2 one, so a noisy Sigma_0
shifts the K levels by several t.

    mpirun -n 8 python plaquette_2x2.py --scheme DCA --U 6 --beta 10
"""
import argparse
import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.gfs import dyson
from triqs.operators import c, dagger
from triqs_cthyb import Solver
from triqs_cthyb.tail_fit import tail_fit

SPINS = ("up", "dn")
K_NAMES = ("G", "X", "Y", "M")
SITES = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
K_VECS = np.pi * SITES  # (0,0), (pi,0), (0,pi), (pi,pi), in K_NAMES order
# P[R, K] = e^{i K.R} / 2: real, symmetric and orthogonal, since every phase is +-1
P = np.cos(SITES @ K_VECS.T) / 2
GF_STRUCT = [(f"{s}_{K}", 1) for s in SPINS for K in K_NAMES]

parser = argparse.ArgumentParser(description="2x2 DCA / CDMFT for the square-lattice Hubbard model")
parser.add_argument("--scheme", choices=("DCA", "CDMFT"), default="DCA")
parser.add_argument("--U", type=float, default=6.0)
parser.add_argument("--t", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=10.0)
parser.add_argument("--mu", type=float, default=None, help="Chemical potential (default U/2, half filling)")
parser.add_argument("--n_loops", type=int, default=10)
parser.add_argument("--n_k", type=int, default=32, help="k~ points per direction of the reduced zone")
parser.add_argument("--n_l", type=int, default=40, help="Legendre coefficients for G_l")
parser.add_argument("--fit_min_w", type=float, default=4.0, help="Sigma is replaced by its fitted tail above this")
parser.add_argument("--fit_max_w", type=float, default=10.0)
parser.add_argument("--n_cycles", type=int, default=100000, help="QMC cycles per rank")
parser.add_argument("--n_warmup_cycles", type=int, default=5000)
parser.add_argument("--length_cycle", type=int, default=100)
args = parser.parse_args()
mu = args.U / 2 if args.mu is None else args.mu


def c_site(s, r):
    return sum(P[r, k] * c(f"{s}_{K}", 0) for k, K in enumerate(K_NAMES))


def n_site(s, r):
    return dagger(c_site(s, r)) * c_site(s, r)


def eps(k):
    return -2 * args.t * (np.cos(k[..., 0]) + np.cos(k[..., 1]))


def lattice_hamiltonians(scheme, n_k):
    """H(k~) in the K basis, shape (n_k^2, 4, 4). The grid is symmetric under both mirrors, so
    the CDMFT average stays diagonal in K to rounding."""
    k1 = (np.arange(n_k) + 0.5) * np.pi / n_k - np.pi / 2
    kt = np.stack(np.meshgrid(k1, k1, indexing="ij"), axis=-1).reshape(-1, 2)
    H = eps(kt[:, None, :] + K_VECS[None, :, :])[:, :, None] * np.eye(4)
    if scheme == "CDMFT":
        V = P.T @ (np.exp(1j * kt @ SITES.T)[:, :, None] * P)
        H = V @ H @ V.conj().transpose(0, 2, 1)
    return H


def coarse_grain(H, iw, sigma):
    """< [iw + mu - H(k~) - Sigma]^-1 >_k~ in the K basis, shape (n_w, 4, 4); sigma is (n_w, 4)."""
    zeta = (iw[:, None] + mu - sigma)[:, :, None] * np.eye(4)
    return sum(np.linalg.inv(zeta - h) for h in H) / len(H)


h_int = args.U * sum(n_site("up", r) * n_site("dn", r) for r in range(len(SITES)))
H = lattice_hamiltonians(args.scheme, args.n_k)

S = Solver(beta=args.beta, gf_struct=GF_STRUCT, n_l=args.n_l)
iw = np.array([complex(w) for w in S.G0_iw[GF_STRUCT[0][0]].mesh.values()])
iw0 = len(iw) // 2

# Start from the Hartree self-energy of the paramagnetic half-filled solution
sigma = np.full((len(iw), len(K_NAMES)), args.U / 2, dtype=complex)

out = f"plaquette_2x2_{args.scheme}_U{args.U:.2f}_beta{args.beta:.1f}_mu{mu:.3f}.h5"
if mpi.is_master_node():
    with HDFArchive(out, "w") as A:
        A["params"] = {k: v for k, v in vars(args).items() if v is not None}
        A["mu"] = mu
        A["K_names"] = list(K_NAMES)

for it in range(args.n_loops):
    G_c = coarse_grain(H, iw, sigma)
    off_diag = np.abs(G_c - G_c * np.eye(4)).max()
    assert off_diag < 1e-10, f"cluster G is not diagonal in K (max |G_KK'| = {off_diag:.2e})"
    G0 = 1 / (1 / np.diagonal(G_c, axis1=1, axis2=2) + sigma)
    for s in SPINS:
        for k, K in enumerate(K_NAMES):
            S.G0_iw[f"{s}_{K}"].data[:, 0, 0] = G0[:, k]

    S.solve(h_int=h_int, n_cycles=args.n_cycles, n_warmup_cycles=args.n_warmup_cycles,
            length_cycle=args.length_cycle, measure_G_l=True,
            measure_density_matrix=True, use_norm_as_weight=True)

    G_iw = S.G0_iw.copy()
    for name, g in G_iw:
        g.set_from_legendre(S.G_l[name])
    Sigma_iw_raw = dyson(G0_iw=S.G0_iw, G_iw=G_iw)
    Sigma_iw = tail_fit(Sigma_iw_raw.copy(), fit_min_w=args.fit_min_w, fit_max_w=args.fit_max_w,
                        fit_max_moment=3, fit_known_moments=S.Sigma_moments)

    # Paramagnetic and C4-symmetric (X = Y)
    sigma = np.mean([[Sigma_iw[f"{s}_{K}"].data[:, 0, 0] for K in K_NAMES] for s in SPINS], axis=0).T
    sigma[:, 1:3] = sigma[:, 1:3].mean(axis=1, keepdims=True)
    for s in SPINS:
        for k, K in enumerate(K_NAMES):
            Sigma_iw[f"{s}_{K}"].data[:, 0, 0] = sigma[:, k]

    density = sum(occ[0, 0].real for occ in S.orbital_occupations.values()) / len(SITES)
    sign = np.real(S.average_sign)
    mpi.report(f"iteration {it}: <n> = {density:.4f}, sign = {sign:.4f}, Im Sigma_K(iw_0) = "
               + ", ".join(f"{K} {sigma[iw0, k].imag:.4f}" for k, K in enumerate(K_NAMES)))

    if mpi.is_master_node():
        with HDFArchive(out, "a") as A:
            A[f"it{it}"] = dict(G0_iw=S.G0_iw, G_iw=G_iw, Sigma_iw=Sigma_iw, Sigma_iw_raw=Sigma_iw_raw,
                                G_tau=S.G_tau, G_l=S.G_l, density=density, average_sign=sign)
            A["n_iter"] = it + 1
