# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""Chemical-potential calibration. Away from half filling mu is found once by bisection and
pinned in the submit script, so every solver runs the same Hamiltonian.

`bisect_mu` takes any non-decreasing `density(mu)`: a short CTSEG run (`ctseg_probe`, the
single-orbital models) or an exact diagonalization (`ed_probe`). With a stochastic probe `tol`
should sit near the probe's statistical error.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

from . import kernels


def _cached(density, samples, label, verbose):
    """`density`, memoized in `samples` and printing each new evaluation."""
    def n_of(mu):
        if mu not in samples:
            samples[mu] = float(density(mu))
            if verbose:
                print(f"  {label} mu = {mu:+.6f} -> n = {samples[mu]:.6f}")
        return samples[mu]
    return n_of


def bracket_mu(density, mu_guess, target_n, step=0.5, max_expand=8, verbose=True):
    """`(mu_lo, mu_hi)` with `density(mu_lo) <= target_n <= density(mu_hi)`, expanding
    geometrically from `mu_guess`, and the dict of every `(mu, n)` evaluated."""
    samples = {}
    n_of = _cached(density, samples, "probe", verbose)

    n_guess = n_of(mu_guess)
    if abs(n_guess - target_n) < 1e-12:
        return (mu_guess, mu_guess), samples

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
    """Bisect `density(mu) = target_n` until `|n - target_n| < tol` (a tolerance on the density,
    the quantity actually known). Returns `(mu, n_achieved, samples)`."""
    (mu_lo, mu_hi), samples = bracket_mu(density, mu_guess, target_n, step=step, verbose=verbose)
    if mu_lo == mu_hi:
        return mu_lo, samples[mu_lo], samples

    n_of = _cached(density, samples, "bisect", verbose)
    n_lo = n_of(mu_lo)
    for _ in range(max_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        n_mid = n_of(mu_mid)
        if abs(n_mid - target_n) < tol:
            break
        if (n_mid - target_n) * (n_lo - target_n) <= 0:
            mu_hi = mu_mid
        else:
            mu_lo, n_lo = mu_mid, n_mid
    return mu_mid, n_mid, samples


def report(mu, n_achieved, target_n, label, variable):
    """The block to paste into a submit script."""
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


def add_calibration_args(parser):
    """The probe knobs of the CTSEG-calibrated benchmarks."""
    parser.add_argument("--target_n", type=float, default=0.75, help="Target density per spin-orbital")
    parser.add_argument("--probe_cycles", type=int, default=20000, help="MC cycles per probe")
    parser.add_argument("--tol", type=float, default=2e-3, help="Tolerance on the density")


def ctseg_probe(Model, args, gf_struct, couplings):
    """`density(mu)`: mean density per spin-orbital of `Model` at `mu`, from a short CTSEG run.

    `couplings(model)` returns `(jperp_tau or None, {(s, s'): D0_ss'})` in CTSEG's convention.
    """
    from triqs.gfs import Fourier
    from triqs_ctseg import Solver

    def density(mu):
        probe = argparse.Namespace(**vars(args))
        probe.mu, probe.filling = mu, args.target_n
        model = Model(probe)
        jperp_tau, d0 = couplings(model)

        S = Solver(gf_struct=gf_struct, beta=model.beta, n_tau=model.n_tau,
                   n_tau_bosonic=model.n_tau_bosonic)
        S.Delta_tau << Fourier(model.delta_iw())
        if jperp_tau is not None:
            S.Jperp_tau << kernels.as_gf(jperp_tau, model.beta)
        for (s1, s2), d in d0.items():
            S.D0_tau[s1, s2] << kernels.as_gf(d, model.beta)

        S.solve(h_int=model.h_int(), h_loc0=model.h_loc0(),
                length_cycle=args.length_cycle,
                n_warmup_cycles=max(args.probe_cycles // 20, 500),
                n_cycles=args.probe_cycles,
                measure_nn_tau=False, measure_F_tau=False, measure_pert_order=False,
                measure_densities=True)
        # CTSEG's direct density, a time average: -G(beta) is too noisy for a short probe.
        return float(np.mean([S.results.densities[bl][i] for bl, size in gf_struct for i in range(size)]))

    return density


def ed_probe(run_ed, model_args):
    """`density(mu)`: mean density per spin-orbital from one exact solve by the script `run_ed`,
    run as a subprocess (it builds its Hamiltonian at import), parsed from its `<n_a> =` line."""
    out_dir = os.path.join(os.path.dirname(run_ed), 'data', 'calib')

    def density(mu):
        out = subprocess.run(
            [sys.executable, run_ed, *model_args, '--mu', repr(float(mu)),
             '--n_ph_check', '0', '--n_tau', '101', '--n_iw', '64', '--out_dir', out_dir],
            capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"run_ed.py failed at mu={mu}:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
        for line in out.stdout.splitlines():
            if line.startswith('<n_a> ='):
                vals = line.split('=', 1)[1].split('(')[0].strip().strip('[]').split()
                return float(np.mean([float(v) for v in vals]))
        raise RuntimeError(f"could not parse <n_a> from run_ed.py output:\n{out.stdout[-2000:]}")

    return density
