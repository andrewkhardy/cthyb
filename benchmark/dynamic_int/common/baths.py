# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""Hybridization functions for the single-orbital benchmarks, defined at any beta.

The baths are particle-hole symmetric (poles at +-eps_k with equal weight V_k^2), so
Re Delta = 0 and mu = U/2 is exactly half filling for a particle-hole symmetric interaction.

`DMFT_BATH_POLES` is the bath of the original spin-spin benchmark, Delta = iw + U/2 - G0^-1
from `dmft_loop/i_001/S/G0_iw/up` (U = 4) of the old `ctint.ref.h5`, now in the
`dynamical_development` repo: a non-negative least-squares fit of -Im Delta, each weight split
evenly between +eps and -eps. `semicircular_delta` is the first-iteration Bethe bath, a much
stronger low-frequency hybridization, so results are not comparable between the two.
"""
from triqs.gfs import Gf, MeshImFreq, SemiCircular, inverse, iOmega_n

# (eps_k, V_k^2)
DMFT_BATH_POLES = (
    (-4.20000000, 0.04214332),
    (-3.28888889, 0.00266141),
    (-2.83333333, 0.19722748),
    (-1.92222222, 0.07571404),
    (-1.46666667, 0.08200148),
    (-1.01111111, 0.05961020),
    (-0.55555556, 0.03876191),
    (-0.10000000, 0.00187684),
    (+0.10000000, 0.00187684),
    (+0.55555556, 0.03876191),
    (+1.01111111, 0.05961020),
    (+1.46666667, 0.08200148),
    (+1.92222222, 0.07571404),
    (+2.83333333, 0.19722748),
    (+3.28888889, 0.00266141),
    (+4.20000000, 0.04214332),
)


def imfreq_mesh(beta, n_iw):
    return MeshImFreq(beta=beta, statistic="Fermion", n_iw=n_iw)


def pole_delta(mesh, poles=DMFT_BATH_POLES, target_shape=(1, 1)):
    r"""`sum_k V_k^2 / (iw - eps_k)` on `mesh`, the same poles on every diagonal entry."""
    delta = Gf(mesh=mesh, target_shape=target_shape)
    delta.zero()
    for orb in range(target_shape[0]):
        for eps, v_sq in poles:
            delta[orb, orb] << delta[orb, orb] + v_sq * inverse(iOmega_n - eps)
    return delta


def semicircular_delta(mesh, half_bandwidth=2.0, target_shape=(1, 1)):
    r"""First-iteration Bethe hybridization `(D/2)^2 * SemiCircular(D)` on every diagonal entry."""
    delta = Gf(mesh=mesh, target_shape=target_shape)
    delta.zero()
    for orb in range(target_shape[0]):
        delta[orb, orb] << (half_bandwidth / 2) ** 2 * SemiCircular(half_bandwidth)
    return delta


def flat_delta(mesh, v_sq=0.49, eps=0.0, target_shape=(1, 1)):
    r"""One bath site per orbital, `V^2 / (iw - eps)`: ED-representable, particle-hole symmetric
    only at eps = 0 (which is what build_delta uses)."""
    return pole_delta(mesh, poles=((eps, v_sq),), target_shape=target_shape)


def build_delta(mesh, kind="dmft", half_bandwidth=2.0, v_sq=0.49, target_shape=(1, 1)):
    """Dispatch on a `--bath` choice: 'dmft', 'semicircular' or 'discrete'."""
    if kind == "dmft":
        return pole_delta(mesh, target_shape=target_shape)
    if kind == "semicircular":
        return semicircular_delta(mesh, half_bandwidth, target_shape=target_shape)
    if kind == "discrete":
        return flat_delta(mesh, v_sq, target_shape=target_shape)
    raise ValueError(f"Unknown bath kind {kind!r}; expected 'dmft', 'semicircular' or 'discrete'")
