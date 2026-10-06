# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""
Chemical-potential calibration. Away from half filling, mu is found once by bisection and pinned
in the submit script, so every solver runs the identical model. `density(mu)` can be a short QMC
probe or an exact ED; n(mu) is monotonic, so bracketing then bisecting is safe. With a noisy
probe, set `tol` near its statistical error.
"""


def bracket_mu(density, mu_guess, target_n, step=0.5, max_expand=8, verbose=True):
    """`(mu_lo, mu_hi)` with density(mu_lo) <= target_n <= density(mu_hi), and every (mu, n) probed."""
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
    """Bisect density(mu) = target_n to within `tol` in the density. Returns (mu, n_achieved, samples)."""
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
    """The line to paste into the submit script."""
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

