# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""Cluster DMFT (DCA or CDMFT) for the single-band Hubbard model on the square lattice, solved by CTHYB: the clusters of
clusters.py (2x2 by default, the 8-site Betts cluster 8A, 4x4, or any superlattice added there),

    H = -t sum_<ij>,s c^dag_is c_js + U sum_i n_i,up n_i,dn - mu sum_i,s n_i,s,

with the Pauli proposal of the pair moves mixed in at --pauli_prob (0.5 by default; neither scheme is in the regime
where it defaults to 1). On 2x2 it sampled the same sign, order and Sigma as 0, with the same (DCA) or half (CDMFT) the
autocorrelation time. A cycle is about one autocorrelation time by default, 250 moves in DCA and 1000 in CDMFT (100-move
cycles gave ~2 and ~9 on 2x2), after 5e5 warmup moves. On 8A a DCA move took ~0.3 s on 8 threads locally: try --probe
first, and pass e.g. --n_warmup_cycles 20 --length_cycle 10 there.

The solver works in one of three real bases of the cluster orbitals, c_R = sum_o Q[R, o] c_o (--basis), and the
blocks of its gf_struct are the orbitals that the coarse-grained G of the Hartree start connects; every iteration checks
that G stays inside them. By default (auto) it takes the waves, or the irreps where their blocks are smaller, which is
CDMFT beyond 2x2.
  waves   K and -K combined into the waves cos(K.R) and sin(K.R) (one wave when K = -K), with h_int the
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

The loop measures only what it needs, G and the density matrix (for the CRM). After it, a final solve of --final_time
seconds at the converged Sigma, saved as the group final, also measures the density-density correlators of the sites:
<n_a(tau) n_b(0)>, a = sigma N + R, of every pair (nn_dlr_sites, by measure_chiAB_tau with chi_A_vec = chi_B_vec the n_a,
in any basis of the solver, as cluster_ctint.py measures them by CTINT's), from it <S^z_Q(tau) S^z_Q(0)> at
Q = (pi, pi) (szsz_Q_dlr), and the equal-time <n_a n_b> twice, to compare their accuracy: exactly from the density
matrix (nn_sites), and as chiAB at tau = 0 (nn_sites_chiAB), the estimator CTINT has. The correlators are measured by
insertion at the nodes of the bosonic DLR grid of --dlr_wmax and --dlr_eps only, and saved as its DLR coefficients, as
G is (G_dlr): no file holds data on a regular tau or Matsubara mesh. It saves the density matrix and the eigenbasis of
h_loc too, for any other equal-time observable (triqs.atom_diag.trace_rho_op, with the operators c_site builds from Q). Every file holds Q and the names of its orbitals, the sites and momenta of the cluster and the git
version of the code.

With eps(k) = -2t (cos kx + cos ky) and the H(k~) of clusters.py on the k~ of its DCA patch,

    G = < [iw + mu - H(k~) - Sigma]^-1 >_k~,   in DCA G(K) = < [iw + mu - eps(K + k~) - Sigma(K)]^-1 >_k~,

and G0^-1 = G^-1 + Sigma. The solver takes Delta = iw - h_loc0 - G0^-1 with the exact h_loc0 = <H(k~)>_k~ - mu, the
1/iw^2 moment of G0 (its Delta interface): the h_loc0 it would extract from the tail of a G0 built from a noisy Sigma is
off by up to ~3t on 2x2 at 2e4 cycles. The loop runs on the frequencies of the DLR Matsubara mesh (--dlr_wmax,
--dlr_eps), on which the solver takes Delta and returns G, the DLR fit of the measured G(tau) (saved as G_dlr too). By
default (--dlr_sigma crm) Sigma is the solver's constrained residual minimization of Dyson's equation with the
density-matrix moments Sigma_0 and Sigma_1, which holds the tail without a fit window; --dlr_sigma dyson feeds back the
plain Dyson Sigma = G0^-1 - G^-1 at each DLR frequency instead, whose noise grows as w^2. Every run saves the plain
Dyson Sigma as Sigma_iw_raw, and, averaged over the spins and the symmetries of the cluster as the Sigma fed back is, in
the sites as Sigma_sites_dyson: in a CRM run, the two estimators of Sigma from the same G (plot_cluster.py, estimators).
--dlr_eps is 1e-7, not 1e-10: on 2x2 DCA at 2e4 cycles the constraint of the CRM is met to ~1e-9 at 1e-7 and missed by
1e3-1e4 at 1e-10, whose basis fits the noise. The output file is tagged with the two and the basis.

Feasibility: CTHYB diagonalizes the cluster Hamiltonian in its invariant subspaces and keeps dense matrices of them.
Before building anything, the script estimates the largest subspace: the half-filled (N_up, N_dn) sector, split evenly
by the symmetries of the cluster (not at all in the site basis). That is a lower bound, 77 states for 8A DCA where CTHYB
finds 660, so a stop on it is safe; it compares one dense matrix of that size with the memory of the node. For 4x4 that
is about 1.3e6 states, 13 TB, and the script stops unless --force. --probe stops after building the atomic problem.

    mpirun -n 8 python cluster_cthyb.py --cluster 2x2 --scheme CDMFT
"""
import argparse
import os
import sys
import time
from math import comb

import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.atom_diag import trace_rho_op
from triqs.gfs import inverse
from triqs.operators import c, dagger
from triqs.operators.util import op_from_block_matrix
from triqs_cthyb import Solver

from clusters import (Cluster, add_model_args, coarse_grain, connected_blocks, dlr_frequencies, gf_on, run_name,
                      start_archive)

SPINS = ("up", "dn")

parser = argparse.ArgumentParser(description="Cluster DCA / CDMFT for the square-lattice Hubbard model, solved by CTHYB")
add_model_args(parser)
parser.add_argument("--basis", choices=("auto", "waves", "irreps", "sites"), default="auto",
                    help="Basis of the solver: waves (Delta diagonal in DCA, U scatters momenta), irreps (the sites combined by "
                         "the symmetries of the scheme), sites (Delta off-diagonal, U local), or auto: the waves, or the irreps "
                         "where their blocks are smaller")
parser.add_argument("--pauli_prob", type=float, default=0.5, help="Probability of the Pauli proposal of the pair moves")
parser.add_argument("--length_cycle", type=int, default=None, help="Moves per cycle, by default 250 in DCA, 1000 in CDMFT")
parser.add_argument("--n_warmup_cycles", type=int, default=None, help="Warmup cycles, by default 5e5 moves")
parser.add_argument("--dlr_eps", type=float, default=1e-7, help="DLR accuracy, matched to the noise of G for the CRM")
parser.add_argument("--dlr_sigma", choices=("crm", "dyson"), default="crm", help="Sigma fed back: CRM or plain Dyson")
parser.add_argument("--probe", action="store_true", help="Only build the atomic problem and report its size")
parser.add_argument("--force", action="store_true", help="Go on even if the memory estimate exceeds the node")
args = parser.parse_args()
mu = args.U / 2 if args.mu is None else args.mu
if args.length_cycle is None:
    args.length_cycle = {"DCA": 250, "CDMFT": 1000}[args.scheme]
if args.n_warmup_cycles is None:
    args.n_warmup_cycles = 500000 // args.length_cycle

# ---- Cluster and lattice ----

CL = Cluster(args.cluster, args.n_k)
N = CL.N
H_SITES = CL.hamiltonians(args.scheme, args.t)  # H(k~)
MESH, iw, iw0 = dlr_frequencies(args.beta, args.dlr_wmax, args.dlr_eps)
mpi.report(f"{args.scheme}: Sigma averaged over {len(CL.perms[args.scheme])} symmetries of the cluster")

# Start from the Hartree self-energy of the paramagnetic half-filled solution
sigma = np.full(len(iw), args.U / 2, dtype=complex)[:, None, None] * np.eye(N)

# ---- Basis of the solver, Q[R, o] with c_R = sum_o Q[R, o] c_o, and its blocks: the orbitals G connects ----

BASES = {"waves": (CL.P, CL.wave_names), "irreps": CL.symmetric_basis(args.scheme),
         "sites": (np.eye(N), [f"R{i}" for i in range(N)])}


def blocks_in(Q):
    return connected_blocks(coarse_grain(Q.T @ H_SITES @ Q, iw + mu, sigma))


if args.basis == "auto":
    size = {basis: max(map(len, blocks_in(BASES[basis][0]))) for basis in ("waves", "irreps")}
    args.basis = "irreps" if size["irreps"] < size["waves"] else "waves"
Q, NAMES = BASES[args.basis]
H_Q = Q.T @ H_SITES @ Q
BLOCKS = blocks_in(Q)
BLOCK_NAMES = ["|".join(NAMES[j] for j in b) for b in BLOCKS]
INSIDE = np.zeros((N, N), dtype=bool)
for b in BLOCKS:
    INSIDE[np.ix_(b, b)] = True
mpi.report(f"{args.scheme} blocks of the {args.basis} basis: {', '.join(BLOCK_NAMES)}")
GF_STRUCT = [(f"{s}_{name}", len(b)) for s in SPINS for name, b in zip(BLOCK_NAMES, BLOCKS)]
WHERE = {o: (name, i) for name, b in zip(BLOCK_NAMES, BLOCKS) for i, o in enumerate(b)}  # orbital -> (block, index)

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
           f"{comb(N, N // 2) ** 2:.2e} states, largest invariant subspace at least ~{largest:.2e} ({split});"
           f" one dense matrix of it ~ {need / 1e9:.3g} GB, node memory {have / 1e9:.3g} GB")
if need > have and not args.force:
    mpi.report("Stopping: the local problem cannot be diagonalized on this node (use --force to try anyway).")
    sys.exit(3)


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


def n_site(s, r):
    return dagger(c_site(s, r)) * c_site(s, r)


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
    G = coarse_grain(H_Q, iw + mu, sigma)
    check_blocks(G)
    G0_iw = S.Delta_iw.copy()
    for name, b in zip(BLOCK_NAMES, BLOCKS):
        g0_inv = np.linalg.inv(block_view(G, b)) + block_view(sigma, b)
        for s in SPINS:
            S.Delta_iw[f"{s}_{name}"].data[:] = iw[:, None, None] * np.eye(len(b)) - H0[np.ix_(b, b)] - g0_inv
            G0_iw[f"{s}_{name}"].data[:] = np.linalg.inv(g0_inv)
    return G0_iw


def symmetrized(sigma_iw):
    """A Sigma in the blocks of the solver (a BlockGf), paramagnetic and averaged over the symmetries of the cluster:
    (n_w, N, N) in the basis of the solver, zero outside its blocks"""
    sigma = np.zeros((len(iw), N, N), dtype=complex)
    for name, b in zip(BLOCK_NAMES, BLOCKS):
        sigma[np.ix_(np.arange(len(iw)), b, b)] = np.mean([sigma_iw[f"{s}_{name}"].data for s in SPINS], axis=0)
    return np.where(INSIDE, Q.T @ CL.symmetrize(Q @ sigma @ Q.T, args.scheme) @ Q, 0)


h_int = args.U * sum(n_site("up", r) * n_site("dn", r) for r in range(N))

params = dict(h_int=h_int, h_loc0=h_loc0, n_cycles=args.n_cycles, n_warmup_cycles=args.n_warmup_cycles,
              length_cycle=args.length_cycle, max_time=args.max_time, pauli_prob=args.pauli_prob, dlr_sigma=args.dlr_sigma,
              measure_density_matrix=True, use_norm_as_weight=True, imag_threshold=1e-10)  # the k~ sum leaves ~1e-13 in Im Delta of CDMFT blocks

# The final solve also measures <n_a(tau) n_b(0)>, a = sigma N + R, for every pair, at the nodes of the bosonic DLR grid
# of the loop's w_max and eps: the cost of the insertions goes with the number of nodes
NN_OPS = [n_site(s, r) for s in SPINS for r in range(N)]
final_params = dict(params, max_time=args.final_time, measure_chiAB_tau=True, chi_A_vec=NN_OPS, chi_B_vec=NN_OPS,
                    dlr_w_max=args.dlr_wmax, dlr_eps=args.dlr_eps)


def equal_time_nn():
    """<n_a n_b>, a = sigma N + R, from the density matrix of the last solve"""
    nn = np.zeros((2 * N, 2 * N))
    for a in range(2 * N):
        for b in range(a, 2 * N):
            nn[a, b] = nn[b, a] = trace_rho_op(S.density_matrix, NN_OPS[a] * NN_OPS[b], S.h_loc_diagonalization).real
    return nn

if args.probe:
    set_delta(sigma)
    t0 = time.time()
    S.solve(**{**params, "n_cycles": 0, "n_warmup_cycles": 0, "perform_post_proc": False})
    mpi.report(f"Atomic problem built in {time.time() - t0:.1f} s")
    sys.exit(0)

out = os.path.join(args.out_dir, run_name(args, mu) + f"_pauli{args.pauli_prob:.2f}_{args.dlr_sigma}_eps{args.dlr_eps:.0e}"
                                                    f"_{args.basis}.h5")
if mpi.is_master_node():
    start_archive(out, args, mu, CL, gf_struct=GF_STRUCT, Q=Q, orbital_names=NAMES)

for it in range(args.n_loops + (args.final_time > 0)):
    final = it == args.n_loops  # the longer solve at the converged Sigma, which measures the correlators
    G0_iw = set_delta(sigma)
    t0 = time.time()
    S.solve(**(final_params if final else params))
    wall = time.time() - t0

    G_iw, Sigma_iw = S.G_iw, S.Sigma_iw.copy()
    Sigma_iw_raw = G0_iw.copy()
    Sigma_iw_raw << inverse(G0_iw) - inverse(G_iw)

    # Paramagnetic, averaged over the symmetries of the cluster, in the blocks; saved as Sigma_iw, and in the sites. The
    # plain Dyson Sigma of the same G, averaged alike, is saved in the sites too, to compare the estimators
    sigma = symmetrized(Sigma_iw)
    for name, b in zip(BLOCK_NAMES, BLOCKS):
        for s in SPINS:
            Sigma_iw[f"{s}_{name}"].data[:] = block_view(sigma, b)
    Sigma_sites = gf_on(MESH, Q @ sigma @ Q.T)
    Sigma_sites_dyson = gf_on(MESH, Q @ symmetrized(Sigma_iw_raw) @ Q.T)

    density = sum(np.trace(occ).real for occ in S.orbital_occupations.values()) / N
    sign = np.real(S.average_sign)
    sig0 = np.diagonal(sigma[iw0])
    dims = np.array(S.h_loc_diagonalization.get_subspace_dims())
    cost = dict(n_measures=S.n_measures, warmup_seconds=S.warmup_time, accumulation_seconds=S.accumulation_time,
                n_ranks=mpi.size, n_subspaces=len(dims), largest_subspace=int(dims.max()))
    mpi.report(f"{'final' if final else f'iteration {it}'}: <n> = {density:.4f}, sign = {sign:.4f}, order = {S.average_order:.1f}, "
               f"autocorrelation {S.auto_corr_time:.2f} cycles, solve {wall:.0f} s, max |Im Sigma(iw_0)| = {np.abs(sig0.imag).max():.4f}, "
               f"{S.n_measures / (mpi.size * S.accumulation_time):.3g} measurements/s per rank, "
               f"{len(dims)} subspaces of at most {dims.max()} states")

    if mpi.is_master_node():
        data = dict(G0_iw=G0_iw, G_iw=G_iw, Sigma_iw=Sigma_iw, Sigma_sites=Sigma_sites, Sigma_iw_raw=Sigma_iw_raw,
                    Sigma_sites_dyson=Sigma_sites_dyson, G_dlr=S.G_dlr, density=density, average_sign=sign,
                    average_order=S.average_order, auto_corr_time=S.auto_corr_time, solve_seconds=wall, **cost)
        if final:  # the equal-time <n_a n_b> twice: from the density matrix, and as chiAB at tau = 0
            nn_dm, nn_chi = equal_time_nn(), S.chiAB_dlr(0.0).real
            mpi.report(f"equal-time <n_a n_b>: max |chiAB(0) - density matrix| = {np.abs(nn_chi - nn_dm).max():.2e}")
            data.update(nn_dlr_sites=S.chiAB_dlr, nn_sites=nn_dm, nn_sites_chiAB=nn_chi, szsz_Q_dlr=CL.szsz_Q(S.chiAB_dlr),
                        density_matrix=S.density_matrix, h_loc_diagonalization=S.h_loc_diagonalization)
        with HDFArchive(out, "a") as A:
            A["final" if final else f"it{it}"] = data
            if not final:
                A["n_iter"] = it + 1
