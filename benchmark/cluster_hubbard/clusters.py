# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""Square-lattice clusters for DCA and CDMFT, shared by cluster_cthyb.py (CTHYB) and cluster_ctint.py (CTINT).

A cluster is given by its superlattice vectors A1, A2: the sites are the lattice points in the cell they span, the
cluster momenta K the points of the reciprocal superlattice (B_i . A_j = 2 pi delta_ij) in the Brillouin zone, and k~
runs over the cell spanned by B1, B2 around 0, which for these clusters is the DCA patch. The real cluster-momentum basis
combines K and -K into the waves cos(K.R) and sin(K.R) (one wave when K = -K).
"""
import numpy as np

SUPERLATTICES = {
    "2x2": ((2, 0), (0, 2)),
    "8A": ((2, 2), (2, -2)),  # Betts 8A: K = (0,0), (pi,0), (0,pi), (pi,pi), (+-pi/2, +-pi/2)
    "4x4": ((4, 0), (0, 4)),
}


def wrap(k):
    """Momenta reduced to [-pi, pi)"""
    return (np.asarray(k) + np.pi) % (2 * np.pi) - np.pi


class Cluster:

    def __init__(self, name, n_k):
        A = np.array(SUPERLATTICES[name], dtype=float)  # rows A1, A2
        self.name = name
        self.N = int(round(abs(np.linalg.det(A))))
        B = 2 * np.pi * np.linalg.inv(A).T  # rows B1, B2

        span = int(np.abs(A).sum())
        pts = np.array([(x, y) for x in range(-span, span + 1) for y in range(-span, span + 1)])
        frac = pts @ np.linalg.inv(A)
        self.sites = pts[np.all((frac > -1e-9) & (frac < 1 - 1e-9), axis=1)]  # R

        ks = wrap(np.array([m * B[0] + n * B[1] for m in range(self.N) for n in range(self.N)]))
        _, first = np.unique(np.round(ks, 9), axis=0, return_index=True)
        self.K = ks[np.sort(first)]
        assert len(self.sites) == self.N and len(self.K) == self.N, f"{name}: {len(self.sites)} sites, {len(self.K)} momenta"

        u = (np.arange(n_k) + 0.5) / n_k - 0.5
        self.k_patch = np.array([a * B[0] + b * B[1] for a in u for b in u])  # k~, symmetric under k~ -> -k~

        self.P, self.K_of, self.wave_names = self._real_basis()

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

    def superlattice_hamiltonian(self, eps, kt):
        """t(k~) in the site basis, t_RR' = 1/N sum_K e^{i (K + k~).(R - R')} eps(K + k~)"""
        phase = np.exp(1j * self.sites @ (self.K + kt).T) / np.sqrt(self.N)  # [R, K]
        return (phase * eps(self.K + kt)) @ phase.conj().T
