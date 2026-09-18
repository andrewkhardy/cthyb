# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""
Chemical-potential calibration for the dynamic_int benchmarks.

Away from half filling there is no closed form for mu, so it is found once by a scan and
then *pinned* in the submit script, so every solver runs the identical model. That matters
more than it might seem: if each solver found its own mu, differences in Sigma would mix
solver disagreement with a different Hamiltonian, and the benchmark would measure nothing.

`bisect_mu` takes any callable `density(mu) -> n`, so it works with a cheap QMC probe
(CTSEG is the natural choice for the single-orbital models: sign-free here and ~14x faster
than CTHYB) or with an exact diagonalization (for the multiorbital benchmarks, where ED
gives `n(mu)` exactly in seconds).

Monotonicity makes this well-posed: `n(mu)` is non-decreasing for any fermionic impurity
model, so bracketing and bisecting is safe. With a stochastic `density` the bisection
cannot converge below the noise, so `tol` should be set near the statistical error of the
probe and `n_probe_cycles` kept small -- there is no point resolving mu to 1e-4 when the
density is only known to 1e-3.
"""
import numpy as np


def bracket_mu(density, mu_guess, target_n, step=0.5, max_expand=8, verbose=True):
    """Find `(mu_lo, mu_hi)` with `density(mu_lo) <= target_n <= density(mu_hi)`.

    Expands outward from `mu_guess` geometrically. Returns the bracket and a dict of every
    `(mu, n)` evaluated, so nothing is wasted and the scan can be inspected afterwards.
    """
    samples = {}

    def n_of(mu):
        if mu not in samples:
            samples[mu] = float(density(mu))
            if verbose:
                print(f"  probe mu = {mu:+.6f} -> n = {samples[mu]:.6f}")
        return samples[mu]

    n_guess = n_of(mu_guess)
    if abs(n_guess - target_n) < 1e-12:
        return (mu_guess, mu_guess), samples

    # Higher mu means higher density, so walk in the direction of the deficit.
    direction = 1.0 if n_guess < target_n else -1.0
    mu_a, n_a = mu_guess, n_guess
    for i in range(max_expand):
        mu_b = mu_guess + direction * step * (2 ** i)
        n_b = n_of(mu_b)
        if (n_a - target_n) * (n_b - target_n) <= 0:
            return (min(mu_a, mu_b), max(mu_a, mu_b)), samples
        mu_a, n_a = mu_b, n_b
    raise RuntimeError(
        f"Could not bracket n = {target_n} after {max_expand} expansions from mu = {mu_guess}; "
        f"probed {sorted(samples.items())}. Is the target filling reachable for this model?")


def bisect_mu(density, target_n, mu_guess=0.0, tol=2e-3, max_iter=20, step=0.5, verbose=True):
    """Bisect `density(mu) = target_n`.

    Returns `(mu, n_achieved, samples)`. `tol` is on the *density*, not on mu, which is the
    quantity actually known; with a stochastic probe set it near the probe's statistical
    error rather than smaller.
    """
    (mu_lo, mu_hi), samples = bracket_mu(density, mu_guess, target_n, step=step, verbose=verbose)
    if mu_lo == mu_hi:
        return mu_lo, samples[mu_lo], samples

    def n_of(mu):
        if mu not in samples:
            samples[mu] = float(density(mu))
            if verbose:
                print(f"  bisect mu = {mu:+.6f} -> n = {samples[mu]:.6f}")
        return samples[mu]

    n_lo, n_hi = n_of(mu_lo), n_of(mu_hi)
    mu_mid, n_mid = 0.5 * (mu_lo + mu_hi), None
    for _ in range(max_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        n_mid = n_of(mu_mid)
        if abs(n_mid - target_n) < tol:
            break
        if (n_mid - target_n) * (n_lo - target_n) <= 0:
            mu_hi, n_hi = mu_mid, n_mid
        else:
            mu_lo, n_lo = mu_mid, n_mid
    return mu_mid, n_mid, samples


def report(mu, n_achieved, target_n, label, variable):
    """The block to paste into a submit script, so the calibration is recorded not retyped."""
    lines = [
        "",
        "=" * 74,
        f"Calibrated mu for {label}",
        f"  target n = {target_n:.4f} per spin-orbital, achieved n = {n_achieved:.6f}",
        f"  deviation = {n_achieved - target_n:+.2e}",
        "",
        f"  Paste into the submit script:   {variable}={mu:.6f}",
        "=" * 74,
        "",
    ]
    return "\n".join(lines)


def interpolate_from_samples(samples, target_n):
    """Best linear estimate of mu at `target_n` from all `(mu, n)` pairs collected.

    Useful with a noisy probe: a straight-line fit through the bracketing points uses every
    evaluation rather than only the last bisection step.
    """
    mus = np.array(sorted(samples))
    ns = np.array([samples[mu] for mu in mus])
    if len(mus) < 2:
        return float(mus[0])
    # Restrict to the two points straddling the target where possible.
    below = np.where(ns <= target_n)[0]
    above = np.where(ns >= target_n)[0]
    if len(below) and len(above):
        i, j = below[-1], above[0]
        if i != j and ns[j] != ns[i]:
            return float(mus[i] + (target_n - ns[i]) * (mus[j] - mus[i]) / (ns[j] - ns[i]))
    slope, intercept = np.polyfit(ns, mus, 1)
    return float(slope * target_n + intercept)
