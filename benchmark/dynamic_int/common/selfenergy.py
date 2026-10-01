# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""The one self-energy convention every solver's G goes through, so a difference between two
curves is solver physics, not bookkeeping:

    Sigma(iw) = G0(iw)^-1 - G(iw)^-1,    G0(iw)^-1 = iw + mu - eps - Delta(iw).

Build G0 from the model's inputs, never from the solver: solve() shifts h_loc in place by the
Lang-Firsov K'(0), so a mu read back afterwards depends on the routing. Do not tail-fit:
S.Sigma_moments sees only the static h_int, not the retarded interaction. At large beta prefer
the Legendre route (sigma_from_G_l) to a Dyson inversion of a noisy G(tau).
"""
import numpy as np
from triqs.gfs import BlockGf, Gf, MeshImFreq, Fourier, LegendreToMatsubara, inverse, iOmega_n


def g0_inverse_iw(mesh, mu, delta_iw, eps=None):
    r"""`iw + mu - eps - Delta(iw)` for one block on `mesh`; `mu` is a scalar or one value per
    orbital, `eps` an optional (n, n) one-body term beyond -mu."""
    n_orb = delta_iw.target_shape[0]
    mu_mat = np.diag(np.full(n_orb, mu, dtype=float) if np.isscalar(mu) else np.asarray(mu, dtype=float))
    out = Gf(mesh=mesh, target_shape=delta_iw.target_shape)
    out << iOmega_n
    out.data[:] += mu_mat
    if eps is not None:
        out.data[:] -= np.asarray(eps, dtype=float)
    out.data[:] -= delta_iw.data
    return out


def sigma_from_G_iw(G_iw, mu, delta_iw, eps=None):
    r"""`Sigma(iw) = G0(iw)^-1 - G(iw)^-1` block by block. `mu`, `delta_iw` and `eps` are either
    shared by every block or a dict / BlockGf keyed by block name."""
    sigma = G_iw.copy()
    for name, g in G_iw:
        g0_inv = g0_inverse_iw(g.mesh, _per_block(mu, name), _per_block(delta_iw, name), _per_block(eps, name))
        sigma[name] << g0_inv - inverse(g)
    return sigma


def _per_block(value, name):
    """`value[name]` for a dict or BlockGf keyed by block, else `value` itself."""
    if isinstance(value, dict):
        return value.get(name)
    if isinstance(value, BlockGf):
        return value[name]
    return value


def G_iw_from_G_tau(G_tau, n_iw):
    """Fourier transform `G(tau) -> G(iw)` on `n_iw` positive frequencies, deliberately without
    tail moments (the solver's ignore the retarded interaction)."""
    beta = G_tau.mesh.beta
    blocks = []
    for _, g in G_tau:
        g_iw = Gf(mesh=MeshImFreq(beta=beta, statistic="Fermion", n_iw=n_iw), target_shape=g.target_shape)
        g_iw << Fourier(g)
        blocks.append(g_iw)
    return BlockGf(name_list=[n for n, _ in G_tau], block_list=blocks)


def G_iw_from_G_l(G_l, n_iw):
    """`G_l -> G(iw)` via `LegendreToMatsubara` (needs `measure_G_l=True`)."""
    beta = G_l.mesh.beta
    blocks = []
    for _, g in G_l:
        g_iw = Gf(mesh=MeshImFreq(beta=beta, statistic="Fermion", n_iw=n_iw), target_shape=g.target_shape)
        g_iw << LegendreToMatsubara(g)
        blocks.append(g_iw)
    return BlockGf(name_list=[n for n, _ in G_l], block_list=blocks)


def sigma_from_G_tau(G_tau, n_iw, mu, delta_iw, eps=None):
    """Dyson self-energy from `G(tau)`, the cross-check route."""
    return sigma_from_G_iw(G_iw_from_G_tau(G_tau, n_iw), mu, delta_iw, eps)


def sigma_from_G_l(G_l, n_iw, mu, delta_iw, eps=None):
    """Dyson self-energy from the Legendre `G_l`, the preferred route."""
    return sigma_from_G_iw(G_iw_from_G_l(G_l, n_iw), mu, delta_iw, eps)


def sigma_from_F_tau(F_tau, G_iw):
    r"""CTSEG's improved estimator `Sigma(iw) = F(iw) / G(iw)` (`measure_F_tau=True`), which uses no
    G0, mu or Delta; triqs_ctseg's postprocess_sigma without the tail fit."""
    from triqs.gfs import make_hermitian, make_zero_tail

    sigma = G_iw.copy()
    sigma.zero()
    for name, g in G_iw:
        f_iw = g.copy()
        f_iw.zero()
        f_iw.set_from_fourier(F_tau[name], make_zero_tail(f_iw, n_moments=1))
        f_iw << make_hermitian(f_iw)
        sigma[name].data[:] = f_iw.data / g.data
    return make_hermitian(sigma)


def density_from_G_iw(G_iw):
    r"""Diagonal densities per spin-orbital from `G(iw).density()`, flat in block order. Lower
    variance than `-G(beta)`, which reads only the noisiest point of the tau grid."""
    out = []
    for _, g in G_iw:
        out.extend(np.diag(g.density()).real)
    return np.array(out)


def diagnose(sigma_values, w_n, mu=None, w_lo=1.0, w_hi=8.0, w_max=20.0):
    r"""One-line checks on `Sigma(iw_n)`, w_n <= w_max: the frequencies with Im Sigma >= 0 (a bug
    at low frequency, truncation noise at high), and Re Sigma, which at half filling is exactly
    mu at every frequency (G0 uses the bare mu, the solver the particle-hole symmetric one)."""
    sigma_values = np.asarray(sigma_values)
    w_n = np.asarray(w_n)

    keep = w_n <= w_max
    sigma_values, w_n = sigma_values[keep], w_n[keep]

    n_bad = int(np.sum(sigma_values.imag >= 0))
    first_bad = float(w_n[sigma_values.imag >= 0][0]) if n_bad else float("inf")

    window = (w_n > w_lo) & (w_n < w_hi)
    re_mean = float(sigma_values[window].real.mean()) if window.any() else float("nan")
    re_std = float(sigma_values[window].real.std()) if window.any() else float("nan")

    text = (f"Sigma(w<={w_max:g}): Im>=0 at {n_bad}/{len(w_n)} freqs"
            + (f" (first w={first_bad:.1f} of {w_n[-1]:.0f})" if n_bad else " -- causal")
            + f"; Re Sigma({w_lo:g}<w<{w_hi:g}) = {re_mean:.4f} +- {re_std:.4f}")
    if mu is not None and np.isscalar(mu):
        text += f"  [exact at half filling: mu = {float(mu):.4f}]"
    return dict(n_noncausal=n_bad, first_noncausal=first_bad, re_mean=re_mean, re_std=re_std, text=text)


def matsubara_frequencies(mesh):
    """Positive Matsubara frequencies of `mesh` as a plain array."""
    return np.array([complex(w).imag for w in mesh if complex(w).imag > 0])


def positive_frequency_part(g_iw, orb=(0, 0)):
    """`(w_n, values)` of one orbital component on the positive Matsubara frequencies."""
    n = len(g_iw.mesh) // 2
    values = g_iw.data[n:, orb[0], orb[1]]
    return matsubara_frequencies(g_iw.mesh), values
