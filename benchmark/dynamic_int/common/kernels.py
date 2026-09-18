# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""
Retarded interaction kernels for the dynamic_int benchmarks, at any beta.

Two kernels, deliberately distinct, because the benchmarks need different things:

`boson_Q` -- one Einstein boson, the closed form used everywhere in this repo

    Q(tau) = -cosh(omega_0 (tau - beta/2)) / (2 omega_0 sinh(omega_0 beta / 2)),

so that integrating out a phonon `H = ... + omega_0 b^dag b + g N (b + b^dag)/sqrt(2 omega_0)`
gives `D(tau) = g^2 Q(tau)` (doc notes, Eqs. holstein and Qtau; same expression as
`test/python/kanamori_dyn.py`). Note `Q < 0`. Exact at any beta with nothing to transfer,
since `omega_0` is a declared model parameter. Used by `hubbard_holstein`,
`kanamori_phonon` and `vb_dimer` -- every benchmark with an ED reference, because one
boson mode per pole is what ED has to represent explicitly.

`spin_kernel_Q` -- the multi-pole retarded spin-spin kernel of the original single-orbital
benchmark, which came from a stored beta = 10 DMFT solution rather than a closed form, and
is *not* a single pole (a one-pole fit leaves 5.8% residual). Note `Q > 0`, the opposite
sign convention to `boson_Q`; `spin_spin/model.py` applies `spin_kernel = -J * Q`, so a
positive `J` is antiferromagnetic. Used by `spin_spin` only.

Transferring the spin kernel between temperatures
-------------------------------------------------
A kernel's tau data is tied to its beta, but its *spectral density* is not, and that is
what transfers. Writing the reference kernel as a non-negative sum of bosonic poles

    Q(tau) = sum_m rho_m K_{omega_m}(tau),
    K_omega(tau) = cosh(omega (tau - beta/2)) / sinh(omega beta / 2),   rho_m >= 0,

is exactly the DLR/spectral representation, and rescaling tau -- evaluating the beta_ref
curve at tau * beta_ref / beta, which is the cheap way to do this with `fit_gf_dlr` -- is
algebraically the same poles with every frequency scaled:

    Q_ref(tau beta_ref/beta) = sum_m rho_m K_{omega_m beta_ref/beta}(tau) at beta.

`spin_kernel_Q` evaluates that directly from the pole table, so it needs no stored tau data
and no DLR machinery at run time.

The amplitude factor matters. Since `int_0^beta K_omega dtau = 2/omega`, scaling the
frequencies scales the static part of the interaction by beta/beta_ref -- and that static
part, K'(0), is what enters mu and the effective static interaction. Raw tau-rescaling to
beta = 100 would take K'(0) from -0.762 to -7.62, i.e. a ten times more strongly coupled
model rather than the same model at a lower temperature. The extra `beta_ref/beta` factor
holds K'(0) fixed while keeping the kernel long-ranged in tau/beta, which is the point of
running beta = 100 at all. (Holding everything fixed is impossible: the alternative of
fixing the frequencies instead decays to Q(beta/2) = 8e-5 at beta = 100 and barely
exercises the retarded sampling.) `test_kernels.py` asserts the invariance.

Provenance of the pole table
----------------------------
Fitted once to `dmft_loop/i_000/Q_tau` of the old `benchmark/dynamic_int/ctint.ref.h5`, by
`fit_spin_kernel_poles.py` in the `dynamical_development` repo, where that h5 now lives.
7 poles reproduce it to 1.5e-3 relative, which is the stored kernel's own Monte-Carlo noise
floor -- the fit residual has 194 sign changes over 2001 points and its second-difference
rms (2.29e-5) equals the data's own (2.32e-5), and refining the frequency grid from 16 to
256 points changes neither the residual nor the 7 active poles. Sum rules reproduce the
reference to 6 digits: int Q dtau = 1.523985 (data 1.523983), K'(0) = -0.761992 (equal).
"""
import numpy as np
from triqs.gfs import Gf, MeshImTime

# Reference temperature the pole table was fitted at.
SPIN_KERNEL_BETA_REF = 10.0

# (omega_m, rho_m) of the reference retarded spin-spin kernel; see "Provenance" above.
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
    r"""cosh(omega (tau - beta/2)) / sinh(omega beta / 2), written so that large
    omega * beta cannot overflow (the 39.8 pole at beta = 100 would otherwise give
    cosh(1990))."""
    decay = np.exp(-beta * omega)
    return (np.exp(-tau * omega) + np.exp(-(beta - tau) * omega)) / (1.0 - decay)


def boson_Q(beta, n_tau, omega_0):
    r"""Single-Einstein-boson kernel as a plain array on `tau_grid(beta, n_tau)`.

        Q(tau) = -cosh(omega_0 (tau - beta/2)) / (2 omega_0 sinh(omega_0 beta / 2))

    Negative, symmetric about beta/2. `D(tau) = g^2 Q(tau)` for a coupling
    `g N (b + b^dag)/sqrt(2 omega_0)`, so K'(0) = g^2 / (2 omega_0^2) exactly
    (`triqs_cthyb.dynamical_interactions.kprime_0_boson`).
    """
    tau = tau_grid(beta, n_tau)
    return -_pole_kernel(omega_0, tau, beta) / (2.0 * omega_0)


def spin_kernel_Q(beta, n_tau, poles=SPIN_KERNEL_POLES, beta_ref=SPIN_KERNEL_BETA_REF):
    r"""The reference retarded spin-spin kernel at `beta`, as a plain array on
    `tau_grid(beta, n_tau)`.

    Positive, symmetric about beta/2, and reduces to the fitted beta = 10 curve when
    `beta == beta_ref`. Built by scaling every pole frequency by `beta_ref / beta` and the
    overall amplitude by the same factor, which keeps `int Q dtau` and `K'(0)` exactly
    beta-independent -- see the module docstring for why that is the right invariant.
    """
    scale = beta_ref / beta
    tau = tau_grid(beta, n_tau)
    total = np.zeros_like(tau)
    for omega, rho in poles:
        total += rho * _pole_kernel(omega * scale, tau, beta)
    return scale * total


def as_gf(data, beta, statistic="Boson", target_shape=()):
    """Wrap a tau-sampled array as a Gf on a matching mesh.

    `target_shape=()` gives the scalar_valued Gf that `add_dyn_vertex` requires;
    `target_shape=(1, 1)` the matrix_valued one that `D0_tau`/`Jperp_tau` take.
    """
    g = Gf(mesh=MeshImTime(beta=beta, statistic=statistic, n_tau=len(data)), target_shape=target_shape)
    if target_shape == ():
        g.data[:] = data
    else:
        g.data[:] = np.asarray(data).reshape((len(data),) + tuple(1 for _ in target_shape))
    return g
