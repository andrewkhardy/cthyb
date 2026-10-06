# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""
One self-energy convention for every solver, so differences between curves are physics:

    Sigma(iw) = G0(iw)^-1 - G(iw)^-1,    G0(iw)^-1 = iw + mu - eps - Delta(iw).

Build G0 from the input mu and Delta, never from the solver's h_loc, which solve() shifts by the
Lang-Firsov static part. Do not tail-fit: the solver's Sigma moments only see the static h_int.
Prefer the Legendre route (sigma_from_G_l) at large beta; sigma_from_G_tau is the cross-check.
"""
import numpy as np
from triqs.gfs import BlockGf, Gf, MeshImFreq, Fourier, LegendreToMatsubara, inverse, iOmega_n


def g0_inverse_iw(mesh, mu, delta_iw=None, eps=None, target_shape=None):
    r"""`G0(iw)^-1 = iw + mu - eps - Delta(iw)` for one block, on `mesh`.

    Parameters
    ----------
    mesh : MeshImFreq
    mu : float or (n,) array
        Chemical potential. A scalar is applied to every orbital; an array gives one value
        per orbital, as the Kanamori benchmarks need.
    delta_iw : Gf or None
        Hybridization of this block. None means an isolated impurity (no bath).
    eps : (n, n) array or None
        Static one-body term of this block beyond -mu (crystal field, patch levels).
    target_shape : tuple
        Required when `delta_iw` is None, to size the block.
    """
    if delta_iw is not None:
        target_shape = delta_iw.target_shape
    if target_shape is None:
        raise ValueError("Pass delta_iw or target_shape so the block size is known")

    n_orb = target_shape[0]
    mu_mat = np.diag(np.full(n_orb, mu, dtype=float) if np.isscalar(mu) else np.asarray(mu, dtype=float))
    if np.shape(mu_mat) != (n_orb, n_orb):
        raise ValueError(f"mu has {np.size(mu)} entries but the block has {n_orb} orbitals")

    out = Gf(mesh=mesh, target_shape=target_shape)
    out << iOmega_n
    out.data[:] += mu_mat
    if eps is not None:
        out.data[:] -= np.asarray(eps, dtype=float)
    if delta_iw is not None:
        # Delta may live on a longer mesh than G; take the overlapping window.
        out.data[:] -= _on_mesh(delta_iw, mesh).data
    return out


def _on_mesh(g, mesh):
    """`g` restricted to `mesh`, which must be a sub-mesh of `g`'s (same beta, fewer points)."""
    if g.mesh == mesh:
        return g
    if abs(g.mesh.beta - mesh.beta) > 1e-12:
        raise ValueError(f"beta mismatch: {g.mesh.beta} vs {mesh.beta}")
    out = Gf(mesh=mesh, target_shape=g.target_shape)
    n_g, n_out = len(g.mesh) // 2, len(mesh) // 2
    if n_out > n_g:
        raise ValueError(f"target mesh has {n_out} positive frequencies, source only {n_g}")
    out.data[:] = g.data[n_g - n_out:n_g + n_out]
    return out


def sigma_from_G_iw(G_iw, mu, delta_iw=None, eps=None):
    r"""`Sigma(iw) = G0(iw)^-1 - G(iw)^-1`, block by block.

    `mu`, `delta_iw` and `eps` may each be a single value/Gf/array applied to every block,
    or a dict keyed by block name.
    """
    sigma = G_iw.copy()
    for name, g in G_iw:
        g0_inv = g0_inverse_iw(g.mesh, _per_block(mu, name), _per_block(delta_iw, name),
                               _per_block(eps, name), target_shape=g.target_shape)
        sigma[name] << g0_inv - inverse(g)
    return sigma


def _per_block(value, name):
    """`value[name]` for a dict keyed by block, else `value` itself."""
    if isinstance(value, dict):
        return value.get(name)
    if isinstance(value, BlockGf):
        return value[name]
    return value


def G_iw_from_G_tau(G_tau, n_iw):
    """Fourier transform `G(tau) -> G(iw)`, without tail moments (they ignore the retarded interaction)."""
    beta = G_tau.mesh.beta
    blocks = []
    for _, g in G_tau:
        g_iw = Gf(mesh=MeshImFreq(beta=beta, statistic="Fermion", n_iw=n_iw), target_shape=g.target_shape)
        g_iw << Fourier(g)
        blocks.append(g_iw)
    return BlockGf(name_list=[n for n, _ in G_tau], block_list=blocks)


def G_iw_from_G_l(G_l, n_iw):
    """`G_l -> G(iw)` via `LegendreToMatsubara`. Needs `measure_G_l=True`."""
    beta = G_l.mesh.beta
    blocks = []
    for _, g in G_l:
        g_iw = Gf(mesh=MeshImFreq(beta=beta, statistic="Fermion", n_iw=n_iw), target_shape=g.target_shape)
        g_iw << LegendreToMatsubara(g)
        blocks.append(g_iw)
    return BlockGf(name_list=[n for n, _ in G_l], block_list=blocks)


def sigma_from_G_tau(G_tau, n_iw, mu, delta_iw=None, eps=None):
    """Dyson self-energy from `G(tau)`. Cross-check route; see `sigma_from_G_l`."""
    return sigma_from_G_iw(G_iw_from_G_tau(G_tau, n_iw), mu, delta_iw, eps)


def sigma_from_G_l(G_l, n_iw, mu, delta_iw=None, eps=None):
    """Dyson self-energy from the Legendre `G_l`. Preferred route."""
    return sigma_from_G_iw(G_iw_from_G_l(G_l, n_iw), mu, delta_iw, eps)


def sigma_from_F_tau(F_tau, G_iw):
    r"""CTSEG's improved estimator, `Sigma(iw) = F(iw) / G(iw)` (needs `measure_F_tau=True`).

    It uses no G0, mu or Delta, so agreement with the Dyson routes checks those conventions.
    Like triqs_ctseg's postprocess_sigma, but without the tail fit.
    """
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
    r"""Diagonal densities per spin-orbital from `G(iw)`, flat, in block order.

    Prefer this to `-G_tau.data[-1]`, which reads the single noisiest tau point.
    """
    out = []
    for _, g in G_iw:
        out.extend(np.diag(g.density()).real)
    return np.array(out)


def diagnose(sigma_values, w_n, mu=None, w_lo=1.0, w_hi=8.0, w_max=20.0):
    r"""Quick checks on `Sigma(iw_n)` for w_n <= w_max, as a one-line report.

    Counts non-causal points (Im Sigma >= 0; expected only near the top of a Fourier-based
    mesh) and averages Re Sigma over w_lo < w < w_hi. At half filling Re Sigma = mu exactly at
    every frequency, so passing `mu` prints that reference.
    """
    sigma_values = np.asarray(sigma_values)
    w_n = np.asarray(w_n)

    # Above ~w_max a Legendre-derived Sigma is truncation noise; pass w_max=inf to see the whole mesh
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
    """Positive Matsubara frequencies of `mesh` as a plain array, for plotting."""
    return np.array([complex(w).imag for w in mesh if complex(w).imag > 0])


def positive_frequency_part(g_iw, orb=(0, 0)):
    """`(w_n, values)` of one orbital component on the positive Matsubara frequencies."""
    n = len(g_iw.mesh) // 2
    values = g_iw.data[n:, orb[0], orb[1]]
    return matsubara_frequencies(g_iw.mesh), values
