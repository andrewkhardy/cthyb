# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""Retarded interaction kernels for the single-orbital benchmarks, at any beta.

`boson_Q`, one Einstein boson (hubbard_holstein; kanamori_phonon and vb_dimer write the same
kernel as cosh/sinh in their model.py):

    Q(tau) = -cosh(omega_0 (tau - beta/2)) / (2 omega_0 sinh(omega_0 beta / 2)) < 0,

so a phonon coupled as `g N (b + b^dag)/sqrt(2 omega_0)` gives D(tau) = g^2 Q(tau).

`spin_kernel_Q`, the retarded spin-spin kernel of the original single-orbital benchmark (> 0;
spin_spin/model.py uses spin_kernel = -J Q): a 7-pole fit, to its Monte-Carlo noise, of
`dmft_loop/i_000/Q_tau` of the old `ctint.ref.h5` (fit_spin_kernel_poles.py in the
`dynamical_development` repo), Q(tau) = sum_m rho_m K_{omega_m}(tau) with
K_omega(tau) = cosh(omega (tau - beta/2)) / sinh(omega beta / 2). Another beta is reached by
scaling every omega_m and the overall amplitude by beta_ref/beta: the shape in tau/beta is
kept and int Q dtau, hence the static part K'(0) that enters mu, is beta-independent.
"""
import numpy as np
from triqs.gfs import Gf, MeshImTime

# Temperature the pole table was fitted at.
SPIN_KERNEL_BETA_REF = 10.0

# (omega_m, rho_m)
SPIN_KERNEL_POLES = (
    (0.09120108393559097, 0.007891594314608433),
    (0.15848931924611140, 0.093178305207964700),
    (0.47863009232263853, 0.016950485781839986),
    (0.83176377110267130, 0.039377860483654964),
    (2.51188643150958240, 0.004667087719818294),
    (4.36515832240166100, 0.012772506324220523),
    (39.81071705534973400, 0.000239013695823047),
)


def tau_grid(beta, n_tau):
    """Uniform tau grid on [0, beta] including both endpoints, matching a bosonic Gf mesh."""
    return np.linspace(0.0, beta, n_tau)


def _pole_kernel(omega, tau, beta):
    r"""cosh(omega (tau - beta/2)) / sinh(omega beta / 2), in a form that cannot overflow at
    large omega beta."""
    decay = np.exp(-beta * omega)
    return (np.exp(-tau * omega) + np.exp(-(beta - tau) * omega)) / (1.0 - decay)


def boson_Q(beta, n_tau, omega_0):
    r"""The single-boson Q(tau) on `tau_grid(beta, n_tau)`; K'(0) of g^2 Q is g^2 / (2 omega_0^2)."""
    tau = tau_grid(beta, n_tau)
    return -_pole_kernel(omega_0, tau, beta) / (2.0 * omega_0)


def spin_kernel_Q(beta, n_tau, poles=SPIN_KERNEL_POLES, beta_ref=SPIN_KERNEL_BETA_REF):
    r"""The reference spin-spin kernel at `beta` on `tau_grid(beta, n_tau)` (module docstring)."""
    scale = beta_ref / beta
    tau = tau_grid(beta, n_tau)
    total = np.zeros_like(tau)
    for omega, rho in poles:
        total += rho * _pole_kernel(omega * scale, tau, beta)
    return scale * total


def as_gf(data, beta):
    """A tau-sampled array as the (1, 1) bosonic Gf that `D0_tau` / `Jperp_tau` take."""
    g = Gf(mesh=MeshImTime(beta=beta, statistic="Boson", n_tau=len(data)), target_shape=(1, 1))
    g.data[:] = np.asarray(data).reshape((len(data), 1, 1))
    return g
