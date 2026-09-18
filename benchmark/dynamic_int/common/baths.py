# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""
Hybridization functions for the dynamic_int benchmarks, defined at any beta.

Every bath here is a sum of discrete poles,

    Delta(iw) = sum_k V_k^2 / (iw - eps_k),

with the poles placed symmetrically, `eps_k` and `-eps_k` carrying equal weight. That
matters for more than tidiness: a symmetric spectral density makes `Re Delta` vanish
identically, which is what makes `mu = U/2` *exactly* half filling for the single-orbital
models (verified: the fitted bath below has `max |Re Delta| = 8.9e-16`). Get this wrong and
the "half filling" runs are silently doped.

Because the poles are fixed numbers rather than tau- or Matsubara-sampled data, evaluating
at a new beta is just evaluating the same formula on the new mesh -- the same physical bath
at a different temperature, with nothing to transfer and no rescaling question. This is
unlike the retarded *kernel* in `kernels.py`, where the amplitude has to be renormalised;
there the static part K'(0) feeds into mu, whereas a particle-hole symmetric bath
contributes nothing to mu at all.

Two baths
---------
`DMFT_BATH_POLES` -- the bath of the original single-orbital spin-spin benchmark, recovered
as poles so that beta = 10 still reproduces the previously validated numbers while beta =
100 becomes available. It is a *correlated* bath: `-Im Delta` is only 0.156 at the first
Matsubara frequency, rises to a maximum of 0.26 around w = 1.6, then falls -- the
pseudo-gapped hybridization of a U = 4 Bethe-lattice DMFT solution near the Mott
transition, not a featureless metallic bath.

`semicircular_delta` -- the first-iteration Bethe bath, `Delta = (D/2)^2 * SemiCircular(D)`,
for when a clean analytically-specified bath is wanted instead. Note it is a much stronger
low-frequency hybridization than the fitted one (`-Im Delta(iw_0) = 0.855` at D = 2 against
0.156), so results are not comparable between the two choices.

Provenance of the fitted bath
-----------------------------
`Delta(iw) = iw + U/2 - G0(iw)^-1` with `U = 4`, from `dmft_loop/i_001/S/G0_iw/up` of the
old `benchmark/dynamic_int/ctint.ref.h5` (now in the `dynamical_development` repo, where
the derivation script also lives). Fitted by non-negative least squares on `-Im Delta`,
using the basis `w/(w^2 + eps^2)`; since that basis is *even* in `eps`, `-Im Delta` alone
cannot distinguish `+eps` from `-eps`, so the fit is done on `|eps|` and each weight split
evenly between the two signs -- which is the correct reconstruction given that `Re Delta`
is measured to be zero, and imposes the symmetry exactly rather than hoping for it.

Accuracy over the full 2050-frequency mesh: `max |Im Delta_fit - Im Delta| = 4.5e-6`
against a scale of 0.261 (1.7e-5 relative), `Re` exact to 8.9e-16, and the first moment
`sum V_k^2 = 0.99999` against the exact value 1 (i.e. `t^2` for half-bandwidth `D = 2`).
"""
import numpy as np
from triqs.gfs import Gf, MeshImFreq, SemiCircular, inverse, iOmega_n

# U the fitted bath was extracted with; only needed to document the extraction, since
# Delta itself is independent of it once recovered.
DMFT_BATH_U = 4.0
DMFT_BATH_BETA_REF = 10.0

# (eps_k, V_k^2), symmetric under eps -> -eps. See "Provenance" above.
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
    r"""`Delta(iw) = sum_k V_k^2 / (iw - eps_k)` on `mesh`, diagonal in the orbital index.

    Every diagonal entry gets the same pole set, which is what the single-orbital and
    degenerate multi-orbital benchmarks want.
    """
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
    r"""First-iteration Bethe hybridization `Delta = (D/2)^2 * SemiCircular(D)`.

    Particle-hole symmetric like `pole_delta`, but a far stronger low-frequency
    hybridization -- see the module docstring.
    """
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
    r"""Single bath site per orbital, `Delta = V^2/(iw - eps)`.

    The hybridization of an ED-representable model: this is what
    `kanamori_phonon` and `vb_dimer --bath discrete` use, so the QMC solver and the exact
    diagonalization see literally the same Hamiltonian. Only particle-hole symmetric when
    `eps == 0`.
    """
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


def bath_first_moment(poles=DMFT_BATH_POLES):
    r"""`sum_k V_k^2`, the coefficient of the `1/iw` tail of `Delta`. A cheap check that a
    pole table was transcribed correctly."""
    return float(sum(v_sq for _, v_sq in poles))
