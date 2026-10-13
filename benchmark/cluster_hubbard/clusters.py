# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""Square-lattice clusters for DCA and CDMFT, and the rest of the loop that cluster_cthyb.py (CTHYB) and
cluster_ctint.py (CTINT) share: the model and loop arguments, the DLR frequencies, the coarse graining, the blocks of G,
and the name and header of the output files.

A cluster is given by its superlattice vectors A1, A2: the sites are the lattice points in the cell they span, the
cluster momenta K the points of the reciprocal superlattice (B_i . A_j = 2 pi delta_ij) in the Brillouin zone, and k~
runs over the cell spanned by B1, B2 around 0, which for these clusters is the DCA patch. The real cluster-momentum basis
combines K and -K into the waves cos(K.R) and sin(K.R) (one wave when K = -K).

The two schemes differ only by the phase of k~ in the lattice Hamiltonian on the cluster, in the site basis

    CDMFT  H(k~)_RR' = 1/N sum_K e^{i (K + k~).(R - R')} eps(K + k~),   the hopping of the superlattice
    DCA    H(k~)_RR' = 1/N sum_K e^{i K.(R - R')} eps(K + k~),           diagonal in K

with eps(k) = -2t (cos kx + cos ky), and the cluster G is < [iw + mu - H(k~) - Sigma]^-1 >_k~ in both (coarse_grain).

and by their symmetries, the site permutations over which Sigma is averaged (symmetrize): in DCA the space group of the
periodic cluster, R -> g R + T modulo the superlattice, with g in the point group of the square that maps the
superlattice onto itself; in CDMFT, whose Sigma is not periodic in the superlattice, only the g that map the cluster onto
itself about its centre. For 2x2 and 4x4 that is the full point group in both schemes (X = Y among others); the 8A
cell, (0,0), (1,-1..1), (2,-1..1), (3,0), has no C4 axis, so there CDMFT keeps only the two mirrors and C2.

symmetric_basis combines the sites into the irreps of these symmetries. The waves are such a basis in DCA and for 2x2
CDMFT; for 8A CDMFT the waves leave Delta one block of 8, the symmetric basis blocks of 3+3+1+1 (A1 and B2 of C2v each
come three times among the 8 sites). No 8-site tile of this superlattice has C4v, and the chiral one with C4,
(0,0), (1,0), (0,1), (1,1), (-1,0), (1,-1), (2,1), (0,2), does no better: 2+2+4 in a real basis.
"""
import os
import subprocess

import numpy as np
from h5 import HDFArchive
from triqs.gfs import Gf, MeshDLRImFreq

SUPERLATTICES = {
    "2x2": ((2, 0), (0, 2)),
    "8A": ((2, 2), (2, -2)),  # Betts 8A: K = (0,0), (pi,0), (0,pi), (pi,pi), (+-pi/2, +-pi/2)
    "4x4": ((4, 0), (0, 4)),
}

# The point group of the square lattice, C4v, as integer matrices acting on (x, y)
POINT_GROUP = [np.array(g) for g in (((1, 0), (0, 1)), ((0, -1), (1, 0)), ((-1, 0), (0, -1)), ((0, 1), (-1, 0)),
                                     ((1, 0), (0, -1)), ((-1, 0), (0, 1)), ((0, 1), (1, 0)), ((0, -1), (-1, 0)))]


def wrap(k):
    """Momenta reduced to [-pi, pi)"""
    return (np.asarray(k) + np.pi) % (2 * np.pi) - np.pi


def dispersion(k, t):
    """eps(k) = -2t (cos kx + cos ky), for k (..., 2)"""
    return -2 * t * (np.cos(k[..., 0]) + np.cos(k[..., 1]))


def coarse_grain(H, zeta, sigma):
    """The cluster G = < [zeta - H(k~) - Sigma]^-1 >_k~, (n_w, N, N), from H (n_k^2, N, N), zeta = iw + mu (n_w,) and
    Sigma (n_w, N, N), all in one basis"""
    z = zeta[:, None, None] * np.eye(H.shape[-1]) - sigma
    return sum(np.linalg.inv(z - h) for h in H) / len(H)


def connected_blocks(G):
    """The blocks of the orbitals that G (n_w, N, N) connects, beyond 1e-8 of its largest element, each sorted"""
    linked = np.abs(G).max(axis=0) > 1e-8 * np.abs(G).max()
    blocks = []
    for j in range(len(linked)):
        if any(j in b for b in blocks):
            continue
        block = {j}
        while (grown := block | set(np.nonzero(linked[sorted(block)].any(axis=0))[0])) != block:
            block = grown
        blocks.append(sorted(block))
    return blocks


def dlr_frequencies(beta, w_max, eps):
    """The symmetrized fermionic DLR Matsubara mesh on which the loop runs, its frequencies iw, and the index of the
    lowest positive one"""
    mesh = MeshDLRImFreq(beta=beta, statistic="Fermion", w_max=w_max, eps=eps, symmetrize=True)
    iw = np.array([complex(w) for w in mesh.values()])
    return mesh, iw, int(np.argmin(np.where(iw.imag > 0, iw.imag, np.inf)))


def gf_on(mesh, data):
    """data (n_w, N, N) as a Gf on mesh"""
    g = Gf(mesh=mesh, target_shape=data.shape[1:])
    g.data[:] = data
    return g


def add_model_args(parser):
    """The arguments of the model, the cluster and the loop that both drivers take. By default every solve runs for
    --max_time seconds, n_cycles being only a cap, so that the two solvers get the same time per iteration."""
    parser.add_argument("--cluster", choices=tuple(SUPERLATTICES), default="2x2")
    parser.add_argument("--scheme", choices=("DCA", "CDMFT"), default="DCA")
    parser.add_argument("--U", type=float, default=6.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--mu", type=float, default=None, help="Chemical potential (default U/2, half filling)")
    parser.add_argument("--n_loops", type=int, default=12, help="Iterations (8 left the 2x2 loop still converging)")
    parser.add_argument("--n_k", type=int, default=16, help="k~ points per direction of the reduced zone")
    parser.add_argument("--dlr_wmax", type=float, default=20.0, help="DLR frequency cutoff")
    parser.add_argument("--n_cycles", type=int, default=10**9, help="QMC cycles per rank, a cap on --max_time")
    parser.add_argument("--max_time", type=int, default=900, help="Seconds per solve, -1 for none")
    parser.add_argument("--final_time", type=int, default=3600,
                        help="Seconds of the final solve at the converged Sigma, which measures the density-density "
                             "correlators; 0 for none")
    parser.add_argument("--out_dir", default=".")


def run_name(args, mu):
    """The start of the name of a run's file, the part plot_cluster.py reads the model from"""
    return f"cluster_{args.cluster}_{args.scheme}_U{args.U:.2f}_beta{args.beta:.1f}_mu{mu:.3f}"


def start_archive(path, args, mu, cluster, **extra):
    """Create the archive of a run with its parameters, mu, the sites and momenta of the cluster, the git version of
    this code, and the extra items. Each iteration then adds a group it<n> and n_iter, and the final solve a group final."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        version = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True,
                                 cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        version = "unknown"
    with HDFArchive(path, "w") as A:
        A["params"] = {k: v for k, v in vars(args).items() if v is not None}
        A["mu"] = mu
        A["sites"], A["K"] = cluster.sites, cluster.K
        A["code_version"] = version
        for key, value in extra.items():
            A[key] = value


class Cluster:

    def __init__(self, name, n_k):
        A = np.array(SUPERLATTICES[name], dtype=float)  # rows A1, A2
        self.name = name
        self.N = int(round(abs(np.linalg.det(A))))
        self.A, self.A_inv = A, np.linalg.inv(A)
        B = 2 * np.pi * self.A_inv.T  # rows B1, B2

        span = int(np.abs(A).sum())
        pts = np.array([(x, y) for x in range(-span, span + 1) for y in range(-span, span + 1)])
        frac = pts @ self.A_inv
        self.sites = pts[np.all((frac > -1e-9) & (frac < 1 - 1e-9), axis=1)]  # R

        ks = wrap(np.array([m * B[0] + n * B[1] for m in range(self.N) for n in range(self.N)]))
        _, first = np.unique(np.round(ks, 9), axis=0, return_index=True)
        self.K = ks[np.sort(first)]
        assert len(self.sites) == self.N and len(self.K) == self.N, f"{name}: {len(self.sites)} sites, {len(self.K)} momenta"

        u = (np.arange(n_k) + 0.5) / n_k - 0.5
        self.k_patch = np.array([a * B[0] + b * B[1] for a in u for b in u])  # k~, symmetric under k~ -> -k~

        self.stagger = 1 - 2 * (self.sites.sum(axis=1) % 2)  # e^{i Q.R} at Q = (pi, pi)

        self.P, self.K_of, self.wave_names = self._real_basis()
        self.perms = {scheme: self._permutations(scheme) for scheme in ("DCA", "CDMFT")}

    def _index_of(self, k):
        return int(np.argmin(np.abs(wrap(self.K - k)).sum(axis=1)))

    def _real_basis(self):
        """P[R, j] real orthogonal with columns cos(K.R) and sin(K.R), the K index of each column, and its name"""
        cols, k_of, names, seen = [], [], [], set()
        for k, K in enumerate(self.K):
            if k in seen:
                continue
            mk = self._index_of(-K)
            phase = self.sites @ K
            label = "K" + "_".join(f"{int(round(4 * c / np.pi)):+d}" for c in K)  # in units of pi/4
            if mk == k:
                cols, k_of, names = cols + [np.cos(phase) / np.sqrt(self.N)], k_of + [k], names + [label]
            else:
                cols += [np.sqrt(2.0 / self.N) * np.cos(phase), np.sqrt(2.0 / self.N) * np.sin(phase)]
                k_of, names = k_of + [k, k], names + [label + "c", label + "s"]
            seen |= {k, mk}
        P = np.array(cols).T
        assert np.allclose(P.T @ P, np.eye(self.N)), "the cos/sin basis is not orthonormal"
        return P, np.array(k_of), names

    def _site_of(self, r):
        """The index of the site equal to r modulo the superlattice"""
        frac = r @ self.A_inv
        r0 = np.rint((frac - np.floor(frac + 1e-9)) @ self.A)
        return int(np.nonzero((self.sites == r0).all(axis=1))[0][0])

    def _permutations(self, scheme):
        """The site permutations of the symmetries of the scheme, one per row: row g holds the index of g(R) for each R"""
        index = {tuple(r): i for i, r in enumerate(self.sites)}
        centre = self.sites.mean(axis=0)
        perms = []
        for g in POINT_GROUP:
            if not np.allclose(self.A @ g.T @ self.A_inv, np.rint(self.A @ g.T @ self.A_inv)):
                continue  # g does not map the superlattice onto itself
            if scheme == "DCA":
                perms += [[self._site_of(g @ r + t) for r in self.sites] for t in self.sites]
                continue
            image = (self.sites - centre) @ g.T + centre
            if np.allclose(image, np.rint(image)) and all(tuple(r) in index for r in np.rint(image).astype(int)):
                perms.append([index[tuple(r)] for r in np.rint(image).astype(int)])
        perms = np.unique(np.array(perms), axis=0)
        assert all(sorted(p) == list(range(self.N)) for p in perms), f"{self.name} {scheme}: a symmetry is not a permutation"
        return perms

    def hamiltonians(self, scheme, t):
        """H(k~) in the site basis for every k~ of the patch, (n_k^2, N, N); see the module docstring"""
        q = self.K[None, :, :] + (self.k_patch[:, None, :] if scheme == "CDMFT" else 0)  # momenta of the phases
        phase = np.exp(1j * np.einsum("rd,tkd->trk", self.sites, np.broadcast_to(q, (len(self.k_patch), self.N, 2))))
        eps = dispersion(self.K[None] + self.k_patch[:, None], t)
        return np.einsum("trk,tk,tsk->trs", phase, eps, phase.conj()) / self.N

    def symmetric_basis(self, scheme):
        """Q[R, o] real orthogonal whose columns combine equivalent sites, each within one isotypic component (the copies of
        one irrep) of the symmetries of the scheme, and their names: a matrix with those symmetries, like Delta, is block
        diagonal in Q, with blocks no larger than the multiplicity of an irrep of dimension one (3+3+1+1 for 8A CDMFT).
        The components are the eigenspaces of a generic invariant matrix grouped by their characters, in decreasing order
        of the characters; within one, the columns are the projections of the sites onto it, orthonormalized in site
        order, so that each column lives on one orbit of sites."""
        R = [np.eye(self.N)[:, p] for p in self.perms[scheme]]  # R_g e_R = e_g(R)
        S = np.random.default_rng(0).normal(size=(self.N, self.N))
        w, V = np.linalg.eigh(self.symmetrize(S + S.T, scheme))
        edges = [0] + [i for i in range(1, self.N) if w[i] - w[i - 1] > 1e-8 * (1 + np.abs(w).max())] + [self.N]
        projectors = {}
        for a, b in zip(edges[:-1], edges[1:]):  # one copy of an irrep per eigenspace, the matrix being generic
            chi = tuple(np.round([np.trace(V[:, a:b].T @ r @ V[:, a:b]) for r in R], 6))
            projectors[chi] = projectors.get(chi, 0) + V[:, a:b] @ V[:, a:b].T
        cols, names = [], []
        for c, chi in enumerate(sorted(projectors, reverse=True)):
            basis = []
            for v in projectors[chi].T:
                v = v - sum(u * (u @ v) for u in basis)
                if np.linalg.norm(v) > 1e-6:
                    basis.append(v / np.linalg.norm(v))
            cols += basis
            names += [f"I{c}{k}" for k in range(len(basis))]
        Q = np.array(cols).T
        assert Q.shape == (self.N, self.N) and np.allclose(Q.T @ Q, np.eye(self.N)), "the symmetric basis is not orthonormal"
        return Q, names

    def szsz_Q(self, nn_tau):
        """<S^z_Q(tau) S^z_Q(0)> at Q = (pi, pi), a scalar Gf, from <n_a(tau) n_b(0)> (a Gf), a = sigma N + R"""
        w = np.concatenate([self.stagger, -self.stagger]) / 2
        g = Gf(mesh=nn_tau.mesh, target_shape=[])
        g.data[:] = np.einsum("a,tab,b->t", w, nn_tau.data, w)
        return g

    def symmetrize(self, m, scheme):
        """m (..., N, N) in the site basis averaged over the symmetries of the scheme, m_RR' -> < m_g(R) g(R') >_g"""
        return np.mean([m[..., p[:, None], p[None, :]] for p in self.perms[scheme]], axis=0)
