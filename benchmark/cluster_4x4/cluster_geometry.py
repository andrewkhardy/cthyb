# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""The cluster of cluster_4x4.py and cluster_ctint.py: an LxL square or a Betts cluster of the square lattice.

A cluster is fixed by its superlattice vectors a1, a2 (the rows of A): N = |det A| sites R, one per superlattice cell,
and N cluster momenta K with K.a in 2 pi Z, K = 2 pi q / D for integer q. The 8-site Betts cluster 8A, a1 = (2, 2) and
a2 = (2, -2), has K = (0, 0), (pi, 0), (0, pi), (pi, pi) and (+-pi/2, +-pi/2), and the full point group of the square
lattice. The k~ of the coarse graining sample the cell of the reciprocal superlattice centred on 0, which is the
Wigner-Seitz cell (the DCA patch) when the reciprocal vectors are orthogonal and of equal length, as for LxL and 8A.

K and -K combine into the real waves cos(K.R) and sin(K.R) (one wave when K = -K), the columns of P.
"""
import itertools
from typing import NamedTuple

import numpy as np

BETTS = {"8A": ((2, 2), (2, -2))}


class Geometry(NamedTuple):
    N: int
    sites: np.ndarray    # (N, 2), R
    K: np.ndarray        # (N, 2), the cluster momenta, in the order of q = (q1, q2) with q1 fastest
    P: np.ndarray        # (N, N), P[R, j] real orthogonal, the waves cos(K.R) and sin(K.R)
    k_of: np.ndarray     # (N,), the K index of each wave
    names: list          # of each wave: K<q1><q2>, with c / s for the two waves of K != -K
    k_tilde: np.ndarray  # (n_k^2, 2)


def superlattice(name):
    if name in BETTS:
        return np.array(BETTS[name])
    L, x, L2 = name.partition("x")
    if not (x and L.isdigit() and L == L2):
        raise ValueError(f"cluster {name!r}: expected LxL or one of {', '.join(BETTS)}")
    return int(L) * np.eye(2, dtype=int)


def geometry(name, n_k):
    """The Geometry of cluster `name` ("LxL" or a key of BETTS), with n_k x n_k points k~"""
    A = superlattice(name)
    N = round(abs(np.linalg.det(A)))
    A_inv = np.linalg.inv(A)
    D = next(d for d in range(1, N + 1) if np.allclose(d * A_inv, np.rint(d * A_inv)))

    # R with fractional coordinates R A^-1 in [0, 1)^2; for LxL that is x, y in [0, L), x fastest
    span = range(-N, N + 1)
    grid = np.array([(x, y) for y in span for x in span])
    frac = grid @ A_inv
    sites = grid[np.all((frac > -1e-9) & (frac < 1 - 1e-9), axis=1)]

    # K = 2 pi A^-1 n over integer n, i.e. q = D A^-1 n mod D
    qs = {tuple(np.rint(D * A_inv @ n).astype(int) % D) for n in itertools.product(range(D), repeat=2)}
    q = np.array(sorted(qs, key=lambda v: (v[1], v[0])))
    assert len(sites) == len(q) == N, f"{len(sites)} sites and {len(q)} momenta for a {N}-site cluster"
    K = 2 * np.pi / D * q
    minus = [next(i for i, v in enumerate(q) if np.all(v == -q[k] % D)) for k in range(N)]

    cols, k_of, names, seen = [], [], [], set()
    for k, (q1, q2) in enumerate(q):
        if k in seen:
            continue
        phase = sites @ K[k]
        if minus[k] == k:
            cols, k_of, names = cols + [np.cos(phase) / np.sqrt(N)], k_of + [k], names + [f"K{q1}{q2}"]
        else:
            cols += [np.sqrt(2 / N) * np.cos(phase), np.sqrt(2 / N) * np.sin(phase)]
            k_of, names = k_of + [k, k], names + [f"K{q1}{q2}c", f"K{q1}{q2}s"]
        seen |= {k, minus[k]}
    P = np.array(cols).T
    assert np.allclose(P.T @ P, np.eye(N)), "the cos/sin basis is not orthonormal"

    G = 2 * np.pi * A_inv.T  # rows: the reciprocal superlattice vectors
    assert abs(G[0] @ G[1]) < 1e-12 and np.isclose(G[0] @ G[0], G[1] @ G[1]), "the k~ cell is not the Wigner-Seitz cell"
    s = (np.arange(n_k) + 0.5) / n_k - 0.5
    k_tilde = np.stack(np.meshgrid(s, s, indexing="ij"), axis=-1).reshape(-1, 2) @ G

    return Geometry(N, sites, K, P, np.array(k_of), names, k_tilde)
