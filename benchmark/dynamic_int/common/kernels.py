# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""
Retarded interaction kernels on a uniform tau grid, at any beta.

boson_Q        One Einstein boson,
                   Q(tau) = -cosh(omega_0 (tau - beta/2)) / (2 omega_0 sinh(omega_0 beta/2)) < 0,
               so a phonon coupled as g N (b + b^dag) / sqrt(2 omega_0) gives D(tau) = g^2 Q(tau).
spin_kernel_Q  The multi-pole spin-spin kernel of the spin_spin benchmark (> 0; the model uses
               -J Q). A sum of bosonic poles fitted at beta_ref = 10,
                   Q(tau) = sum_m rho_m cosh(omega_m (tau - beta/2)) / sinh(omega_m beta/2),
               moved to another beta by scaling every omega_m and the amplitude by beta_ref/beta.
               That keeps the shape in tau/beta and the static part K'(0) fixed (test_kernels.py).

The 7-pole table reproduces the original stored beta = 10 kernel to 1.5e-3, its Monte Carlo
noise floor (fit_spin_kernel_poles.py in the dynamical_development repo).
"""
import numpy as np
from triqs.gfs import Gf, MeshImTime

# Reference temperature the pole table was fitted at.
SPIN_KERNEL_BETA_REF = 10.0

# (omega_m, rho_m) of the reference spin-spin kernel
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
    r"""cosh(omega (tau - beta/2)) / sinh(omega beta / 2), in a form that cannot overflow."""
    decay = np.exp(-beta * omega)
    return (np.exp(-tau * omega) + np.exp(-(beta - tau) * omega)) / (1.0 - decay)


def boson_Q(beta, n_tau, omega_0):
    r"""Q(tau) = -cosh(omega_0 (tau - beta/2)) / (2 omega_0 sinh(omega_0 beta/2)) on `tau_grid(beta, n_tau)`."""
    tau = tau_grid(beta, n_tau)
    return -_pole_kernel(omega_0, tau, beta) / (2.0 * omega_0)


def spin_kernel_Q(beta, n_tau, poles=SPIN_KERNEL_POLES, beta_ref=SPIN_KERNEL_BETA_REF):
    r"""The spin-spin kernel at `beta` on `tau_grid(beta, n_tau)`; frequencies and amplitude scaled by beta_ref/beta."""
    scale = beta_ref / beta
    tau = tau_grid(beta, n_tau)
    total = np.zeros_like(tau)
    for omega, rho in poles:
        total += rho * _pole_kernel(omega * scale, tau, beta)
    return scale * total


def as_gf(data, beta, statistic="Boson", target_shape=()):
    """Wrap a tau-sampled array as a Gf; target_shape=(1, 1) for D0_tau/Jperp_tau inputs."""
    g = Gf(mesh=MeshImTime(beta=beta, statistic=statistic, n_tau=len(data)), target_shape=target_shape)
    if target_shape == ():
        g.data[:] = data
    else:
        g.data[:] = np.asarray(data).reshape((len(data),) + tuple(1 for _ in target_shape))
    return g
