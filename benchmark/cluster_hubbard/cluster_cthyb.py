# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""Cluster DMFT (DCA or CDMFT) for the single-band Hubbard model on the square lattice, solved by CTHYB: the clusters of
clusters.py (2x2 by default, the 8-site Betts cluster 8A, 4x4, or any superlattice added there),

    H = -t sum_<ij>,s c^dag_is c_js + U sum_i n_i,up n_i,dn - mu sum_i,s n_i,s,

with the Pauli proposal of the pair moves mixed in at --pauli_prob (0.5 by default; neither scheme is in the regime
where it defaults to 1).

The solver works in one of three real bases of the cluster orbitals, c_R = sum_o Q[R, o] c_o (--basis), and the
blocks of its gf_struct are the orbitals that the coarse-grained G of the Hartree start connects; every iteration checks
that G stays inside them.
  waves   (default) K and -K combined into the waves cos(K.R) and sin(K.R) (one wave when K = -K), with h_int the
          momentum-conserving scattering of U, as in plaquette_2x2.py. In DCA, G0, Delta and Sigma are diagonal in it by
          construction (G(K) = G(-K) by inversion); in CDMFT only by the point group, which on 2x2 still leaves blocks
          of size 1, but on 8A one block of all 8 waves per spin. For 8A DCA the invariant subspaces CTHYB finds in this
          basis have at most 660 states.
  irreps  the sites combined into the irreps of the symmetries of the scheme (clusters.py), each orbital on one orbit
          of sites: the finest blocks the symmetries allow, 3+3+1+1 per spin for 8A CDMFT, whose tile has only C2v.
  sites   one block of N sites per spin, Delta_RR' off-diagonal, and the local U n_up n_dn: CTHYB can then split the
          atomic problem only by (N_up, N_dn). (The 2x2 CDMFT sign is ~0 in it, the 2x2 DCA sign ~0.01.)
Every iteration saves the measurements, the warmup and accumulation time and the invariant subspaces of the solve, to
compare the cost of the bases.

Every iteration averages Sigma over the spins and over the symmetries of the cluster (clusters.py): in DCA the space
group of the periodic cluster, which leaves one Sigma(K) per star of K (X = Y on 2x2), shared by the cos and sin waves;
in CDMFT the point operations that map the cluster onto itself. Sigma_iw, in the blocks of the solver, is the one fed
back, and Sigma_sites the same in the site basis.

With eps(k) = -2t (cos kx + cos ky) and the H(k~) of clusters.py on the k~ of its DCA patch,

    G = < [iw + mu - H(k~) - Sigma]^-1 >_k~,   in DCA G(K) = < [iw + mu - eps(K + k~) - Sigma(K)]^-1 >_k~,

and G0^-1 = G^-1 + Sigma. The solver takes Delta = iw - h_loc0 - G0^-1 with the exact h_loc0 = <H(k~)>_k~ - mu, the
1/iw^2 moment of G0 (its Delta interface): the h_loc0 it would extract from the tail of a G0 built from a noisy Sigma is
off by up to ~3t on 2x2 at 2e4 cycles. The loop runs on the frequencies of the DLR Matsubara mesh (--dlr_wmax,
--dlr_eps), on which the solver takes Delta and returns G, the DLR fit of the measured G(tau) (saved as G_dlr too). By
default (--dlr_sigma crm) Sigma is the solver's constrained residual minimization of Dyson's equation with the
density-matrix moments Sigma_0 and Sigma_1, which holds the tail without a fit window; --dlr_sigma dyson feeds back the
plain Dyson Sigma = G0^-1 - G^-1 at each DLR frequency instead, whose noise grows as w^2. Every run saves the plain
Dyson Sigma as Sigma_iw_raw. --dlr_eps is 1e-7, not 1e-10: on 2x2 DCA at 2e4 cycles the constraint of the CRM is met to
~1e-9 at 1e-7 and missed by 1e3-1e4 at 1e-10, whose basis fits the noise. The output file is tagged with the two and the
basis.

Feasibility: CTHYB diagonalizes the cluster Hamiltonian in its invariant subspaces and keeps dense matrices of them.
Before building anything, the script estimates the largest subspace: the half-filled (N_up, N_dn) sector, split evenly
by the symmetries of the cluster (not at all in the site basis; an optimistic split, 8A DCA has 660 against 77), and
compares one dense matrix of that size with the memory of the node. For 4x4 that is about 1.3e6 states, 13 TB, and the
script stops unless --force. --probe stops after building the atomic problem.

    mpirun -n 8 python cluster_cthyb.py --cluster 8A --scheme DCA --pauli_prob 0.5
"""
import argparse
import os
import sys
import time
from math import comb

import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.gfs import Gf, MeshDLRImFreq, inverse
from triqs.operators import c, dagger
from triqs.operators.util import op_from_block_matrix
from triqs_cthyb import Solver

from clusters import SUPERLATTICES, Cluster

SPINS = ("up", "dn")

parser = argparse.ArgumentParser(description="Cluster DCA / CDMFT for the square-lattice Hubbard model, solved by CTHYB")
parser.add_argument("--cluster", choices=tuple(SUPERLATTICES), default="2x2")
parser.add_argument("--scheme", choices=("DCA", "CDMFT"), default="DCA")
parser.add_argument("--basis", choices=("waves", "irreps", "sites"), default="waves",
                    help="Basis of the solver: waves (Delta diagonal in DCA, U scatters momenta), irreps (the sites combined by "
                         "the symmetries of the scheme: the finest CDMFT blocks) or sites (Delta off-diagonal, U local)")
parser.add_argument("--U", type=float, default=6.0)
parser.add_argument("--t", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=10.0)
parser.add_argument("--mu", type=float, default=None, help="Chemical potential (default U/2, half filling)")
parser.add_argument("--pauli_prob", type=float, default=0.5, help="Probability of the Pauli proposal of the pair moves")
parser.add_argument("--n_loops", type=int, default=10)
parser.add_argument("--n_k", type=int, default=16, help="k~ points per direction of the reduced zone")
parser.add_argument("--dlr_wmax", type=float, default=20.0, help="DLR frequency cutoff (CTINT's of cluster_ctint.py)")
parser.add_argument("--dlr_eps", type=float, default=1e-7, help="DLR accuracy, matched to the noise of G for the CRM")
parser.add_argument("--dlr_sigma", choices=("crm", "dyson"), default="crm", help="Sigma fed back: CRM or plain Dyson")
parser.add_argument("--n_cycles", type=int, default=100000, help="QMC cycles per rank")
parser.add_argument("--n_warmup_cycles", type=int, default=5000)
parser.add_argument("--length_cycle", type=int, default=100)
parser.add_argument("--max_time", type=int, default=-1, help="Seconds per solve, -1 for none")
parser.add_argument("--probe", action="store_true", help="Only build the atomic problem and report its size")
parser.add_argument("--force", action="store_true", help="Go on even if the memory estimate exceeds the node")
parser.add_argument("--out_dir", default=".")
args = parser.parse_args()
mu = args.U / 2 if args.mu is None else args.mu
CL = Cluster(args.cluster, args.n_k)
N = CL.N

# ---- Feasibility, before any operator is built ----

n_sym = 1 if args.basis == "sites" else len(CL.perms[args.scheme])
largest = comb(N, N // 2) ** 2 / n_sym
need = 8.0 * largest**2
try:
    have = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
except (ValueError, OSError, AttributeError):
    have = float("nan")
split = f"split by the {n_sym} symmetries of the cluster" if n_sym > 1 else "not split by symmetry in the site basis"
mpi.report(f"{args.cluster} cluster, {2 * N} spin-orbitals, {2.0 ** (2 * N):.2e} states; half-filled (N_up, N_dn) sector "
           f"{comb(N, N // 2) ** 2:.2e} states, largest invariant subspace ~ {largest:.2e} ({split});"
           f" one dense matrix of it ~ {need / 1e9:.3g} GB, node memory {have / 1e9:.3g} GB")
if need > have and not args.force:
    mpi.report("Stopping: the local problem cannot be diagonalized on this node (use --force to try anyway).")
    sys.exit(3)

# ---- Lattice, and the basis of the solver: Q[R, o] with c_R = sum_o Q[R, o] c_o ----

if args.basis == "waves":
    Q, NAMES = CL.P, CL.wave_names
elif args.basis == "irreps":
    Q, NAMES = CL.symmetric_basis(args.scheme)
else:
    Q, NAMES = np.eye(N), [f"R{i}" for i in range(N)]


def eps(k):
    return -2 * args.t * (np.cos(k[..., 0]) + np.cos(k[..., 1]))


MESH = MeshDLRImFreq(beta=args.beta, statistic="Fermion", w_max=args.dlr_wmax, eps=args.dlr_eps, symmetrize=True)
iw = np.array([complex(w) for w in MESH.values()])
iw0 = int(np.argmin(np.where(iw.imag > 0, iw.imag, np.inf)))  # the lowest positive frequency

H_Q = Q.T @ CL.hamiltonians(eps, args.scheme) @ Q  # H(k~) in the basis of the solver
mpi.report(f"{args.scheme}: Sigma averaged over {len(CL.perms[args.scheme])} symmetries of the cluster")


def coarse_grain(sigma):
    """The cluster G, (n_w, N, N), from Sigma, both in the basis of the solver: < [iw + mu - H(k~) - Sigma]^-1 >_k~"""
    zeta = (iw + mu)[:, None, None] * np.eye(N) - sigma
    return sum(np.linalg.inv(zeta - h) for h in H_Q) / len(H_Q)


# Start from the Hartree self-energy of the paramagnetic half-filled solution
sigma = np.full(len(iw), args.U / 2, dtype=complex)[:, None, None] * np.eye(N)
G = coarse_grain(sigma)

# ---- Blocks: the orbitals connected by the coarse-grained G ----

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
INSIDE = np.zeros((N, N), dtype=bool)
for b in BLOCKS:
    INSIDE[np.ix_(b, b)] = True
mpi.report(f"{args.scheme} blocks of the {args.basis} basis: {', '.join(BLOCK_NAMES)}")
GF_STRUCT = [(f"{s}_{name}", len(b)) for s in SPINS for name, b in zip(BLOCK_NAMES, BLOCKS)]
WHERE = {o: (name, i) for name, b in zip(BLOCK_NAMES, BLOCKS) for i, o in enumerate(b)}  # orbital -> (block, index)


def block_view(m, b):
    return m[np.ix_(np.arange(len(iw)), b, b)]


def check_blocks(G):
    """Stop if the coarse-grained G has weight outside the blocks"""
    off = np.abs(G[:, ~INSIDE]).max(initial=0.0)
    if off > 1e-8 * np.abs(G).max():
        mpi.report(f"Stopping: the coarse-grained G leaves the blocks (max |G_oo'| outside = {off:.2e}).")
        sys.exit(4)


def c_site(s, r):
    return sum(Q[r, o] * c(f"{s}_{WHERE[o][0]}", WHERE[o][1]) for o in range(N) if abs(Q[r, o]) > 1e-14)


S = Solver(beta=args.beta, gf_struct=GF_STRUCT, delta_interface=True, dlr_wmax=args.dlr_wmax, dlr_eps=args.dlr_eps)
assert S.dlr_mesh == MESH

# h_loc0 = <H(k~)>_k~ - mu
H0 = H_Q.mean(axis=0)
assert np.abs(H0.imag).max() < 1e-10
H0 = H0.real - mu * np.eye(N)
h0_blocks = np.empty(len(GF_STRUCT), dtype=object)
h0_blocks[:] = [H0[np.ix_(b, b)] for s in SPINS for b in BLOCKS]
h_loc0 = op_from_block_matrix(h0_blocks, GF_STRUCT)


def set_delta(sigma):
    """Coarse-grain Sigma, check that G stays in the blocks, and set S.Delta_iw = iw - h_loc0 - G0^-1 with
    G0^-1 = G^-1 + Sigma; G0 as a BlockGf"""
    G = coarse_grain(sigma)
    check_blocks(G)
    G0_iw = S.Delta_iw.copy()
    for name, b in zip(BLOCK_NAMES, BLOCKS):
        g0_inv = np.linalg.inv(block_view(G, b)) + block_view(sigma, b)
        for s in SPINS:
            S.Delta_iw[f"{s}_{name}"].data[:] = iw[:, None, None] * np.eye(len(b)) - H0[np.ix_(b, b)] - g0_inv
            G0_iw[f"{s}_{name}"].data[:] = np.linalg.inv(g0_inv)
    return G0_iw


h_int = args.U * sum(dagger(c_site("up", r)) * c_site("up", r) * dagger(c_site("dn", r)) * c_site("dn", r) for r in range(N))

params = dict(h_int=h_int, h_loc0=h_loc0, n_cycles=args.n_cycles, n_warmup_cycles=args.n_warmup_cycles,
              length_cycle=args.length_cycle, max_time=args.max_time, pauli_prob=args.pauli_prob, dlr_sigma=args.dlr_sigma,
              measure_density_matrix=True, use_norm_as_weight=True, imag_threshold=1e-10)  # the k~ sum leaves ~1e-13 in Im Delta of CDMFT blocks

if args.probe:
    set_delta(sigma)
    t0 = time.time()
    S.solve(**{**params, "n_cycles": 0, "n_warmup_cycles": 0, "perform_post_proc": False})
    mpi.report(f"Atomic problem built in {time.time() - t0:.1f} s")
    sys.exit(0)

out = os.path.join(args.out_dir, f"cluster_{args.cluster}_{args.scheme}_U{args.U:.2f}_beta{args.beta:.1f}_mu{mu:.3f}"
                                 f"_pauli{args.pauli_prob:.2f}_{args.dlr_sigma}_eps{args.dlr_eps:.0e}_{args.basis}.h5")
if mpi.is_master_node():
    with HDFArchive(out, "w") as A:
        A["params"] = {k: v for k, v in vars(args).items() if v is not None}
        A["mu"] = mu
        A["gf_struct"] = GF_STRUCT

for it in range(args.n_loops):
    G0_iw = set_delta(sigma)
    t0 = time.time()
    S.solve(**params)
    wall = time.time() - t0

    G_iw, Sigma_iw = S.G_iw, S.Sigma_iw.copy()
    Sigma_iw_raw = G0_iw.copy()
    Sigma_iw_raw << inverse(G0_iw) - inverse(G_iw)

    # Paramagnetic, averaged over the symmetries of the cluster, in the blocks; saved as Sigma_iw, and in the sites
    sigma = np.zeros((len(iw), N, N), dtype=complex)
    for name, b in zip(BLOCK_NAMES, BLOCKS):
        sigma[np.ix_(np.arange(len(iw)), b, b)] = np.mean([Sigma_iw[f"{s}_{name}"].data for s in SPINS], axis=0)
    sigma = np.where(INSIDE, Q.T @ CL.symmetrize(Q @ sigma @ Q.T, args.scheme) @ Q, 0)
    for name, b in zip(BLOCK_NAMES, BLOCKS):
        for s in SPINS:
            Sigma_iw[f"{s}_{name}"].data[:] = block_view(sigma, b)
    Sigma_sites = Gf(mesh=MESH, target_shape=(N, N))
    Sigma_sites.data[:] = Q @ sigma @ Q.T

    density = sum(np.trace(occ).real for occ in S.orbital_occupations.values()) / N
    sign = np.real(S.average_sign)
    sig0 = np.diagonal(sigma[iw0])
    dims = np.array(S.h_loc_diagonalization.get_subspace_dims())
    cost = dict(n_measures=S.n_measures, warmup_seconds=S.warmup_time, accumulation_seconds=S.accumulation_time,
                n_ranks=mpi.size, n_subspaces=len(dims), largest_subspace=int(dims.max()))
    mpi.report(f"iteration {it}: <n> = {density:.4f}, sign = {sign:.4f}, order = {S.average_order:.1f}, "
               f"autocorrelation {S.auto_corr_time:.2f} cycles, solve {wall:.0f} s, max |Im Sigma(iw_0)| = {np.abs(sig0.imag).max():.4f}, "
               f"{S.n_measures / (mpi.size * S.accumulation_time):.3g} measurements/s per rank, "
               f"{len(dims)} subspaces of at most {dims.max()} states")

    if mpi.is_master_node():
        with HDFArchive(out, "a") as A:
            A[f"it{it}"] = dict(G0_iw=G0_iw, G_iw=G_iw, Sigma_iw=Sigma_iw, Sigma_sites=Sigma_sites, Sigma_iw_raw=Sigma_iw_raw,
                                G_tau=S.G_tau, G_dlr=S.G_dlr, density=density, average_sign=sign, average_order=S.average_order,
                                auto_corr_time=S.auto_corr_time, solve_seconds=wall, **cost)
            A["n_iter"] = it + 1
