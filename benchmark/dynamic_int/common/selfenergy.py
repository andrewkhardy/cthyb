# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""
One self-energy convention for every solver in the dynamic_int benchmarks.

The point of this module is that `Sigma` is computed the *same* way from every solver's
`G`, so that a difference between two curves is solver physics and not bookkeeping.

    Sigma(iw) = G0(iw)^-1 - G(iw),    G0(iw)^-1 = iw + mu - eps - Delta(iw)

Two traps this exists to avoid
------------------------------
1. **Build G0 from the inputs, never from solver internals.** `solve()` mutates `h_loc` in
   place by the Lang-Firsov K'(0) shift, so after a dynamical solve
   `S.h_loc() != h_int + h_loc0` and any mu read back out is route-dependent (it differs
   between `lang_firsov=True` and `False` for the same physical model). The `Delta_iw` and
   `mu` passed here must be the ones that went *in*.

2. **Do not tail-fit.** `S.Sigma_moments` comes from
   `sigma_high_frequency_moments(density_matrix, h_loc_diag, gf_struct, h_int)`, which sees
   only the *static* `h_int` -- the retarded interaction's contribution to the Hartree and
   first moment is simply absent, so a fit anchored on those moments is anchored on the
   wrong asymptote. Compare Sigma on the Matsubara points directly instead (Re and Im
   separately), and take `S.Sigma_iw_raw` if reading it off the solver.

Which G to use
--------------
At beta = 100 a Dyson inversion of a noisy `G(tau)` is unusable at high frequency. Prefer
the Legendre `G_l` route (`sigma_from_G_l`, needing `measure_G_l=True`), which filters the
noise before the inversion, and keep the `G_tau` route as a cross-check. `run_*.py` saves
both under distinct keys so the plots can show the pair.
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
    """Fourier transform `G(tau) -> G(iw)` on `n_iw` positive frequencies.

    No tail/moment information is supplied, deliberately: the only moments available from
    the solver ignore the retarded interaction (see the module docstring), so feeding them
    in would impose a wrong asymptote. That makes the high-frequency end noisy -- which is
    exactly why `sigma_from_G_l` is the preferred route at large beta.
    """
    beta = G_tau.mesh.beta
    blocks = []
    for _, g in G_tau:
        g_iw = Gf(mesh=MeshImFreq(beta=beta, statistic="Fermion", n_iw=n_iw), target_shape=g.target_shape)
        g_iw << Fourier(g)
        blocks.append(g_iw)
    return BlockGf(name_list=[n for n, _ in G_tau], block_list=blocks)


def G_iw_from_G_l(G_l, n_iw):
    """`G_l -> G(iw)` via `LegendreToMatsubara`.

    The Legendre coefficients are a smooth, noise-filtered representation of the same
    measurement, so this is the route to prefer before a Dyson inversion, especially at
    beta = 100. Needs `measure_G_l=True` in the solve.
    """
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
    r"""CTSEG's improved estimator, `Sigma(iw) = F(iw) / G(iw)`.

    `F_tau` is the correlator `-<T c(tau) (interaction term) c^dag(0)>` that CTSEG
    accumulates when `measure_F_tau=True`; dividing by `G` gives Sigma with far better
    high-frequency behaviour than a Dyson inversion, because the noisy denominator cancels.

    Worth noting for cross-solver comparison: this route involves **no** `G0`, `mu` or
    `Delta` at all, so it is completely independent of the chemical-potential bookkeeping
    that `sigma_from_G_iw` depends on. Agreement between the two is therefore a real check
    on the conventions, not a tautology. Mirrors
    `triqs_ctseg.postprocessing.postprocess_sigma`'s improved-estimator branch, but without
    its tail fit (see the module docstring for why fitting is wrong here).
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
    r"""Diagonal densities per spin-orbital from `G(iw)`, as a flat array in block order.

    Prefer this over `-G_tau.data[-1]`. Both are correct in the limit of infinite
    statistics -- `-G(beta)` was verified against `G_iw.density()` to 1e-15 on an exactly
    known non-interacting case -- but `-G(beta)` reads a *single* point of the tau grid,
    the noisiest one, whereas `density()` uses the whole function plus its tail.

    That difference is not academic: a 20k-cycle CTSEG probe gave densities from `-G(beta)`
    scattered by +-0.15, enough to make the measured `n(mu)` non-monotonic (mu = 4.5 gave a
    larger n than mu = 6.0), which no fermionic model permits and which broke a bisection.
    Where a solver measures the density directly -- CTSEG's `measure_densities=True` ->
    `results.densities` -- that is better still, being a plain time average.
    """
    out = []
    for _, g in G_iw:
        out.extend(np.diag(g.density()).real)
    return np.array(out)


def diagnose(sigma_values, w_n, mu=None, w_lo=1.0, w_hi=8.0, w_max=20.0):
    r"""Cheap physical checks on a computed `Sigma(iw_n)`, as a one-line report.

    `causal` counts frequencies with `Im Sigma >= 0`. Some violation at the top of the mesh
    is normal for any Fourier-based route and is not a bug; a violation at *low* frequency
    is. Observed hierarchy at beta = 10, 200k cycles, single-orbital spin-spin: the Legendre
    route stayed causal over all 200 frequencies, CTSEG's improved estimator broke only
    above w = 59, the Dyson routes from G(tau) above w = 27 and 20.

    `re_mean` tests an exact identity available at half filling. Building `G0` with the bare
    `mu` while the solver internally shifts to `mu_eff = U_eff/2` (the Lang-Firsov K'(0)
    shift, which is what makes the effective model particle-hole symmetric) gives

        Sigma = (mu - mu_eff) + Sigma_eff,   Re Sigma_eff = U_eff/2   at p-h symmetry
              => Re Sigma(iw) = mu,   exactly, at every frequency.

    So passing `mu` prints the deviation from it. This is a genuine convention check: it
    fails if the static shift, the level shift's sign, or mu itself is wrong.
    """
    sigma_values = np.asarray(sigma_values)
    w_n = np.asarray(w_n)

    # Only judge causality where Sigma is meaningful. A Legendre-derived Sigma is limited
    # by the number of coefficients: with n_l = 30 the series has no support much beyond
    # w ~ 5, so every frequency above that is truncation noise and flagging it says nothing
    # about the solver. The plots use w <= 15, so w_max = 20 covers the range that matters
    # with room to spare. Pass w_max=inf to inspect the whole mesh deliberately.
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
    """One orbital component of a block on the positive Matsubara frequencies only.

    Returns `(w_n, values)`, so a plot can show Re and Im against `w_n` without
    re-deriving the mesh each time.
    """
    n = len(g_iw.mesh) // 2
    values = g_iw.data[n:, orb[0], orb[1]]
    return matsubara_frequencies(g_iw.mesh), values
