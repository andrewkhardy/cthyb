# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""
Hybridization functions as sums of discrete poles, Delta(iw) = sum_k V_k^2 / (iw - eps_k), so
the same bath can be evaluated at any beta.

DMFT_BATH_POLES     The spin_spin benchmark's bath: a U = 4 Bethe-lattice DMFT solution near the
                    Mott transition (pseudo-gapped, -Im Delta(iw_0) = 0.156), fitted to 1.7e-5.
                    Poles come in +-eps pairs of equal weight, so Re Delta = 0 and mu = U/2 is
                    exactly half filling.
semicircular_delta  First-iteration Bethe bath (D/2)^2 SemiCircular(D); far stronger at low
                    frequency (-Im Delta(iw_0) = 0.855 at D = 2), so not comparable to the above.
flat_delta          One bath site per orbital, V^2 / (iw - eps), which ED can solve.
"""
import numpy as np
from triqs.gfs import Gf, MeshImFreq, SemiCircular, inverse, iOmega_n

# (eps_k, V_k^2), symmetric under eps -> -eps
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
    r"""`Delta(iw) = sum_k V_k^2 / (iw - eps_k)` on `mesh`, the same on every diagonal entry."""
    delta = Gf(mesh=mesh, target_shape=target_shape)
    delta.zero()
    n_orb = target_shape[0] if target_shape else 1
    for orb in range(n_orb):
        for eps, v_sq in poles:
            if target_shape:
                delta[orb, orb] << delta[orb, orb] + v_sq * inverse(iOmega_n - eps)
            else:
                delta << delta + v_sq * inverse(iOmega_n - eps)
    return delta


def semicircular_delta(mesh, half_bandwidth=2.0, target_shape=(1, 1)):
    r"""First-iteration Bethe hybridization `Delta = (D/2)^2 * SemiCircular(D)`."""
    delta = Gf(mesh=mesh, target_shape=target_shape)
    delta.zero()
    n_orb = target_shape[0] if target_shape else 1
    for orb in range(n_orb):
        if target_shape:
            delta[orb, orb] << (half_bandwidth / 2) ** 2 * SemiCircular(half_bandwidth)
        else:
            delta << (half_bandwidth / 2) ** 2 * SemiCircular(half_bandwidth)
    return delta


def flat_delta(mesh, v_sq=0.49, eps=0.0, target_shape=(1, 1)):
    r"""One bath site per orbital, `Delta = V^2/(iw - eps)`; particle-hole symmetric only for eps = 0."""
    return pole_delta(mesh, poles=((eps, v_sq),), target_shape=target_shape)


def build_delta(mesh, kind="dmft", half_bandwidth=2.0, v_sq=0.49, eps_bath=0.0, target_shape=(1, 1)):
    """Dispatch on a `--bath` command-line choice: 'dmft', 'semicircular' or 'discrete'."""
    if kind == "dmft":
        return pole_delta(mesh, target_shape=target_shape)
    if kind == "semicircular":
        return semicircular_delta(mesh, half_bandwidth, target_shape=target_shape)
    if kind == "discrete":
        return flat_delta(mesh, v_sq, eps_bath, target_shape=target_shape)
    raise ValueError(f"Unknown bath kind {kind!r}; expected 'dmft', 'semicircular' or 'discrete'")

