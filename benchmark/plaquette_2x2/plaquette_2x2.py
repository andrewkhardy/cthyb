# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""2x2 cluster DMFT (DCA or CDMFT) for the single-band Hubbard model on the square lattice,

    H = -t sum_<ij>,s c^dag_is c_js - t' sum_<<ij>>,s c^dag_is c_js + U sum_i n_i,up n_i,dn - mu sum_i,s n_i,s,

at a fixed mu (--mu, by default U/2, which is half filling at t' = 0) or at the mu that holds --density electrons per site.

The impurity is solved in the cluster-momentum basis

    c_K,s = 1/2 sum_R e^{-i K.R} c_R,s,    K = G (0,0), X (pi,0), Y (0,pi), M (pi,pi),

in which the Weiss field is diagonal. The four K states are the joint eigenvectors of the two
mirror planes of the plaquette (equivalently, of the 2x2 cluster translations), with four
distinct pairs of eigenvalues, so a one-body quantity with those symmetries has no K != K'
elements: G0, Delta and Sigma, in DCA by construction and in CDMFT by the point group of the
superlattice; t' keeps all of them. gf_struct is therefore 8 blocks of size 1. The cost moves into
h_int, where the Hubbard term becomes the momentum-conserving

    U/4 sum_{K,K',Q} c^dag_{K+Q,up} c_{K,up} c^dag_{K'-Q,dn} c_{K',dn},

built below by rotating the site operators. Autopartition finds N_up, N_dn and the total K.

Self-consistency, with eps(k) = -2t (cos kx + cos ky) - 4t' cos kx cos ky and k~ in the reduced zone [-pi/2, pi/2)^2:

    DCA    G(K) = < [iw + mu - eps(K + k~) - Sigma(K)]^-1 >_k~
    CDMFT  G    = < [iw + mu - t(k~) - Sigma]^-1 >_k~,   t(k~) = V(k~) diag(eps(K + k~)) V(k~)^dag

where V(k~) = P^T diag(e^{i k~.R}) P mixes the K states, so the CDMFT average is diagonal only
by symmetry; that is checked every iteration. Then G0(K)^-1 = G(K)^-1 + Sigma(K), and the solver
takes Delta(K) = iw - h_K - G0(K)^-1 with the exact level h_K = <H(k~)>_KK - mu (its Delta
interface): extracted from the tail of a G0 built from a noisy Sigma, h_K is off by several t.
Sigma is averaged over the spins and over X and Y. The average sign is markedly lower in CDMFT
than in DCA (~0.84 against ~0.998 at the defaults).

Filling: with --density, mu is set before every solve so that the coarse-grained G at the current
Sigma holds the density, -2 sum_K G_K(beta^-) / 4 from the DLR expansion of G on the DLR frequencies
(next_mu). A Sigma from too few measurements (a CRM Sigma that grows with frequency, as after a short
test run) breaks the sum rule G(0+) + G(beta-) = -1 of that G, and then mu steps by the impurity
filling of the last solve instead: (target - n) / kappa, kappa the free compressibility.

The loop runs on the frequencies of the DLR Matsubara mesh (--dlr_wmax, --dlr_eps), on which
the solver takes G0 and returns G, the DLR fit of the measured G(tau). Sigma is the solver's
constrained residual minimization of Dyson's equation (--dlr_sigma crm), which holds the
density-matrix moments Sigma_0 (Hartree) and Sigma_1 fixed. The plain Dyson Sigma, saved as
Sigma_iw_raw and fed back with --dlr_sigma dyson, has noise growing as w^2 at the high DLR
frequencies, and a noisy Sigma_0 shifts the K levels by several t. --dlr_eps is 1e-7, matched to
the noise of G: at 1e-10 the CRM misses its constraint by orders of magnitude.

Monte Carlo: the pair moves mix in the Pauli proposal at --pauli_prob (0.5: on the 2x2 cluster it
samples the same sign, order and Sigma as 0, with the same (DCA) or half (CDMFT) autocorrelation
time). A cycle is about one autocorrelation time, 250 moves in DCA and 1000 in CDMFT, after 5e5
warmup moves, and every solve runs --max_time seconds (--n_cycles only caps it).

After --n_loops iterations, a final solve of --final_time seconds at the converged Sigma, the group
final, also measures <n_a(tau) n_b(0)>, a = 4 s + R, by insertion at the nodes of the bosonic DLR
grid (measure_chiAB_tau). B is only n_up at R = (0, 0): every site of the plaquette is equivalent
(the mirrors act on the sites as R -> R + e mod 2), and with spin flip unfold() completes the other
columns; the insertions cost in proportion to the B operators. From it: the equal-time <n_a n_b>
(also exactly from the density matrix), and <S^z_Q(tau) S^z_Q(0)> at Q = (pi, pi). Its own Sigma,
from fewer measurements while it measures the correlators, is saved but not fed back.

    mpirun -n 8 python plaquette_2x2.py --scheme DCA --U 6 --beta 10
    mpirun -n 8 python plaquette_2x2.py --scheme DCA --tp -0.3 --density 0.9
"""
import argparse

import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from scipy.optimize import brentq
from scipy.special import expit
from triqs.atom_diag import trace_rho_op
from triqs.gfs import Gf, inverse, make_gf_dlr
from triqs.operators import c, dagger, n
from triqs_cthyb import Solver

SPINS = ("up", "dn")
K_NAMES = ("G", "X", "Y", "M")
SITES = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # site index x + 2y
K_VECS = np.pi * SITES  # (0,0), (pi,0), (0,pi), (pi,pi), in K_NAMES order
# P[R, K] = e^{i K.R} / 2: real, symmetric and orthogonal, since every phase is +-1
P = np.cos(SITES @ K_VECS.T) / 2
GF_STRUCT = [(f"{s}_{K}", 1) for s in SPINS for K in K_NAMES]
STAGGER = np.array([1, -1, -1, 1])  # e^{i Q.R} at Q = (pi, pi)

parser = argparse.ArgumentParser(description="2x2 DCA / CDMFT for the square-lattice Hubbard model")
parser.add_argument("--scheme", choices=("DCA", "CDMFT"), default="DCA")
parser.add_argument("--U", type=float, default=6.0)
parser.add_argument("--t", type=float, default=1.0)
parser.add_argument("--tp", type=float, default=0.0, help="Next-nearest-neighbour hopping t'")
parser.add_argument("--beta", type=float, default=10.0)
parser.add_argument("--density", type=float, default=None,
                    help="Electrons per site to hold, by setting mu before every solve (default: fixed --mu)")
parser.add_argument("--mu", type=float, default=None,
                    help="Chemical potential, fixed, or the first guess of --density (default U/2: half filling at t' = 0)")
parser.add_argument("--n_loops", type=int, default=12)
parser.add_argument("--n_k", type=int, default=32, help="k~ points per direction of the reduced zone")
parser.add_argument("--dlr_wmax", type=float, default=20.0, help="DLR frequency cutoff")
parser.add_argument("--dlr_eps", type=float, default=1e-7, help="DLR accuracy, matched to the noise of G for the CRM")
parser.add_argument("--dlr_sigma", choices=("crm", "dyson"), default="crm", help="Sigma fed back: CRM or plain Dyson")
parser.add_argument("--pauli_prob", type=float, default=0.5, help="Probability of the Pauli proposal of the pair moves")
parser.add_argument("--length_cycle", type=int, default=None, help="Moves per cycle, by default 250 in DCA, 1000 in CDMFT")
parser.add_argument("--n_warmup_cycles", type=int, default=None, help="Warmup cycles, by default 5e5 moves")
parser.add_argument("--n_cycles", type=int, default=10**9, help="QMC cycles per rank, a cap on --max_time")
parser.add_argument("--max_time", type=int, default=900, help="Seconds per solve of the loop")
parser.add_argument("--final_time", type=int, default=3600,
                    help="Seconds of the final solve, which measures the correlators; 0 for none")
args = parser.parse_args()
mu = args.U / 2 if args.mu is None else args.mu
if args.length_cycle is None:
    args.length_cycle = {"DCA": 250, "CDMFT": 1000}[args.scheme]
if args.n_warmup_cycles is None:
    args.n_warmup_cycles = 500000 // args.length_cycle


def c_site(s, r):
    return sum(P[r, k] * c(f"{s}_{K}", 0) for k, K in enumerate(K_NAMES))


def n_site(s, r):
    return dagger(c_site(s, r)) * c_site(s, r)


def eps(k):
    cx, cy = np.cos(k[..., 0]), np.cos(k[..., 1])
    return -2 * args.t * (cx + cy) - 4 * args.tp * cx * cy


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


def coarse_grain(H, mu, sigma):
    """< [iw + mu - H(k~) - Sigma]^-1 >_k~ in the K basis, shape (n_w, 4, 4); sigma is (n_w, 4)."""
    zeta = (iw[:, None] + mu - sigma)[:, :, None] * np.eye(4)
    return sum(np.linalg.inv(zeta - h) for h in H) / len(H)


def dlr_expansion(G):
    """The DLR expansion of G (n_w, 4, 4) given on the DLR frequencies"""
    g = Gf(mesh=S.dlr_mesh, target_shape=[4, 4])
    g.data[:] = G
    return make_gf_dlr(g)


def lattice_density(mu, sigma):
    """Electrons per site, both spins, of the coarse-grained G: -2 tr G(beta^-) / 4"""
    return -2 * np.trace(dlr_expansion(coarse_grain(H, mu, sigma))(args.beta)).real / 4


def next_mu(sigma, mu, n_impurity, tol=1e-3):
    """`(mu, how)`: the mu that holds --density in the coarse-grained G at sigma, when that G is physical (its sum rule
    G(0+) + G(beta-) = -1 holds to tol); otherwise mu + (density - n_impurity) / kappa from the impurity filling of the
    last solve, with kappa the free compressibility at the Hartree shift U n / 2 (larger than the interacting one, so
    the step does not overshoot), or mu unchanged before any solve"""
    g = dlr_expansion(coarse_grain(H, mu, sigma))
    violation = np.abs(np.diagonal(g(0.0) + g(args.beta)).real + 1).max()
    if violation < tol:
        excess = lambda m: lattice_density(m, sigma) - args.density
        lo, hi = mu - 0.5, mu + 0.5
        for _ in range(8):
            if excess(lo) <= 0:
                break
            lo -= hi - lo
        for _ in range(8):
            if excess(hi) >= 0:
                break
            hi += hi - lo
        if excess(lo) <= 0 <= excess(hi):
            return brentq(excess, lo, hi, xtol=1e-10), "lattice"
        reason = f"no mu between {lo:g} and {hi:g} holds the density"
    else:
        reason = f"lattice G off its sum rule by {violation:.1e}"
    if n_impurity is None:
        return mu, f"kept: {reason}"
    E = np.linalg.eigvalsh(H).ravel() + args.U * args.density / 2
    fill = lambda m: 2 * expit(args.beta * (m - E)).mean()
    f = expit(args.beta * (brentq(lambda m: fill(m) - args.density, E.min() - 50, E.max() + 50) - E))
    kappa = 2 * args.beta * (f * (1 - f)).mean()
    return mu + (args.density - n_impurity) / kappa, f"impurity step: {reason}"


def unfold(chi):
    """<n_a(tau) n_b(0)> for all a, b = 4 s + R, (n, 8, 8), from the column b = (up, (0, 0)), (n, 8, 1): the mirror
    R -> R + R' mod 2 takes R' to (0, 0), and spin flip takes s' = dn to up"""
    out = np.zeros(chi.shape[:1] + (8, 8), dtype=chi.dtype)
    for s2 in range(2):
        for r2 in range(4):
            for s in range(2):
                for r in range(4):
                    d = (SITES[r] + SITES[r2]) % 2
                    out[:, 4 * s + r, 4 * s2 + r2] = chi[:, 4 * (s ^ s2) + d[0] + 2 * d[1], 0]
    return out


h_int = args.U * sum(n_site("up", r) * n_site("dn", r) for r in range(len(SITES)))
H = lattice_hamiltonians(args.scheme, args.n_k)
H_K = np.diagonal(H.mean(axis=0)).real  # the K levels of <H(k~)>_k~

S = Solver(beta=args.beta, gf_struct=GF_STRUCT, delta_interface=True, dlr_wmax=args.dlr_wmax, dlr_eps=args.dlr_eps)
iw = np.array([complex(w) for w in S.dlr_mesh.values()])
iw0 = int(np.argmin(np.where(iw.imag > 0, iw.imag, np.inf)))  # the lowest positive frequency

params = dict(h_int=h_int, n_cycles=args.n_cycles, n_warmup_cycles=args.n_warmup_cycles, length_cycle=args.length_cycle,
              max_time=args.max_time, pauli_prob=args.pauli_prob, dlr_sigma=args.dlr_sigma,
              measure_density_matrix=True, use_norm_as_weight=True, imag_threshold=1e-10)
NN_OPS = [n_site(s, r) for s in SPINS for r in range(len(SITES))]
final_params = dict(params, max_time=args.final_time, measure_chiAB_tau=True, chi_A_vec=NN_OPS,
                    chi_B_vec=[n_site("up", 0)], dlr_w_max=args.dlr_wmax, dlr_eps=args.dlr_eps)

# Start from the paramagnetic Hartree self-energy U n / 2 at the filling held (half filling without --density)
sigma = np.full((len(iw), len(K_NAMES)), args.U * (1.0 if args.density is None else args.density) / 2, dtype=complex)

filling = f"_n{args.density:.3f}" if args.density is not None else f"_mu{mu:.3f}"
out = (f"plaquette_2x2_{args.scheme}_U{args.U:.2f}" + (f"_tp{args.tp:.2f}" if args.tp != 0 else "")
       + f"_beta{args.beta:.1f}{filling}_{args.dlr_sigma}.h5")
if mpi.is_master_node():
    with HDFArchive(out, "w") as A:
        A["params"] = {k: v for k, v in vars(args).items() if v is not None}
        A["mu"] = mu
        A["K_names"] = list(K_NAMES)

n_impurity, how = None, "fixed"  # the impurity filling of the last solve, and how mu was set
for it in range(args.n_loops + (args.final_time > 0)):
    final = it == args.n_loops  # the longer solve at the converged Sigma, which measures the correlators
    if args.density is not None:
        mu, how = next_mu(sigma, mu, n_impurity)
    G_c = coarse_grain(H, mu, sigma)
    off_diag = np.abs(G_c - G_c * np.eye(4)).max()
    assert off_diag < 1e-10, f"cluster G is not diagonal in K (max |G_KK'| = {off_diag:.2e})"
    n_lattice = lattice_density(mu, sigma)
    h_K = H_K - mu  # the K levels of h_loc0
    h_loc0 = sum(h_K[k] * n(f"{s}_{K}", 0) for s in SPINS for k, K in enumerate(K_NAMES))
    g0_inv = 1 / np.diagonal(G_c, axis1=1, axis2=2) + sigma
    G0_iw = S.Delta_iw.copy()
    for s in SPINS:
        for k, K in enumerate(K_NAMES):
            S.Delta_iw[f"{s}_{K}"].data[:, 0, 0] = iw - h_K[k] - g0_inv[:, k]
            G0_iw[f"{s}_{K}"].data[:, 0, 0] = 1 / g0_inv[:, k]

    S.solve(**(final_params if final else params), h_loc0=h_loc0)

    G_iw, Sigma_iw = S.G_iw, S.Sigma_iw.copy()
    Sigma_iw_raw = G0_iw.copy()
    Sigma_iw_raw << inverse(G0_iw) - inverse(G_iw)

    # Paramagnetic and C4-symmetric (X = Y); the final solve's Sigma is saved, not fed back
    sigma_new = np.mean([[Sigma_iw[f"{s}_{K}"].data[:, 0, 0] for K in K_NAMES] for s in SPINS], axis=0).T
    sigma_new[:, 1:3] = sigma_new[:, 1:3].mean(axis=1, keepdims=True)
    for s in SPINS:
        for k, K in enumerate(K_NAMES):
            Sigma_iw[f"{s}_{K}"].data[:, 0, 0] = sigma_new[:, k]
    if not final:
        sigma = sigma_new

    density = n_impurity = sum(occ[0, 0].real for occ in S.orbital_occupations.values()) / len(SITES)
    sign = np.real(S.average_sign)
    rate = S.n_measures / (mpi.size * S.accumulation_time)
    mpi.report(f"{'final' if final else f'iteration {it}'}: mu = {mu:.5f} ({how}; lattice n = {n_lattice:.5f}), "
               f"<n> = {density:.4f}, sign = {sign:.4f}, order = {S.average_order:.1f}, "
               f"autocorrelation {S.auto_corr_time:.2f} cycles, {rate:.3g} measurements/s per rank, Im Sigma_K(iw_0) = "
               + ", ".join(f"{K} {sigma_new[iw0, k].imag:.4f}" for k, K in enumerate(K_NAMES)))

    if mpi.is_master_node():
        data = dict(G0_iw=G0_iw, G_iw=G_iw, Sigma_iw=Sigma_iw, Sigma_iw_raw=Sigma_iw_raw, G_tau=S.G_tau, G_dlr=S.G_dlr,
                    density=density, lattice_density=n_lattice, mu=mu, mu_from=how, average_sign=sign,
                    average_order=S.average_order, auto_corr_time=S.auto_corr_time, n_measures=S.n_measures,
                    warmup_seconds=S.warmup_time, accumulation_seconds=S.accumulation_time, n_ranks=mpi.size)
        if final:  # every column of <n_a(tau) n_b(0)>; the equal-time <n_a n_b> from the density matrix too
            nn_dlr = Gf(mesh=S.chiAB_dlr.mesh, target_shape=[8, 8])
            nn_dlr.data[:] = unfold(S.chiAB_dlr.data)
            nn_dm = np.array([[trace_rho_op(S.density_matrix, a * b, S.h_loc_diagonalization).real for b in NN_OPS]
                              for a in NN_OPS])
            w = np.concatenate([STAGGER, -STAGGER]) / 2
            szsz = Gf(mesh=nn_dlr.mesh, target_shape=[])
            szsz.data[:] = np.einsum("a,tab,b->t", w, nn_dlr.data, w)
            mpi.report(f"equal-time <n_a n_b>: max |chiAB(0) - density matrix| = {np.abs(nn_dlr(0.0).real - nn_dm).max():.2e}")
            data.update(nn_dlr_sites=nn_dlr, nn_sites=nn_dm, szsz_Q_dlr=szsz)
        with HDFArchive(out, "a") as A:
            A["final" if final else f"it{it}"] = data
            if not final:
                A["n_iter"] = it + 1
