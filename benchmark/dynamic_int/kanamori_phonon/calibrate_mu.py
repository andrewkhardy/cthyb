# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""Find the mu giving a target density for the Kanamori+phonon model, to pin in
run_kanamori_phonon.sh so ED and CTHYB solve the identical Hamiltonian.

Unlike the single-orbital benchmarks, the probe here is *exact*: the ED solves the model
in a few seconds, so n(mu) carries no statistical error and the bisection converges to
machine precision rather than to a noise floor. No tolerance fudging, and no question of
whether a non-monotonic scan is physics or noise.

    python calibrate_mu.py --beta 10  --target_n 0.75
    python calibrate_mu.py --beta 100 --target_n 0.75

Paste the printed value into MU_B10_N075 / MU_B100_N075.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import calibrate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def add_args(parser):
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--U', type=float, default=2.0)
    parser.add_argument('--J', type=float, default=0.3)
    parser.add_argument('--V', type=float, default=0.7)
    parser.add_argument('--eps_bath', type=float, default=0.0)
    parser.add_argument('--omega_0', type=float, default=1.0)
    parser.add_argument('--g', type=float, nargs=2, default=[0.7, 0.3])
    parser.add_argument('--n_ph', type=int, default=24)
    parser.add_argument('--target_n', type=float, default=0.75,
                        help='Target density per spin-orbital (0.5 is half filling)')
    parser.add_argument('--tol', type=float, default=1e-6,
                        help='Tolerance on the density. ED is exact, so this can be tight')


parser = argparse.ArgumentParser(description='Calibrate mu for the Kanamori+phonon benchmark')
add_args(parser)
args = parser.parse_args()

BASE = ['--beta', str(args.beta), '--U', str(args.U), '--J', str(args.J), '--V', str(args.V),
        '--eps_bath', str(args.eps_bath), '--omega_0', str(args.omega_0),
        '--g', str(args.g[0]), str(args.g[1]), '--n_ph', str(args.n_ph)]


def density(mu):
    """Mean density per spin-orbital from one exact ED solve at this mu.

    Runs run_ed.py as a subprocess and reads the <n_a> line it already prints, rather than
    importing it -- run_ed.py builds its Hamiltonian at import time from argparse, so it is
    a script, not a library. A solve is a few seconds, so the process overhead is irrelevant.
    """
    out = subprocess.run(
        [sys.executable, os.path.join(HERE, 'run_ed.py'), *BASE, '--mu', repr(float(mu)),
         '--n_ph_check', '0', '--n_tau', '101', '--n_iw', '64',
         '--out_dir', os.path.join(HERE, 'data', 'calib')],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"run_ed.py failed at mu={mu}:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    for line in out.stdout.splitlines():
        if line.startswith('<n_a> ='):
            vals = line.split('=', 1)[1].split('(')[0].strip().strip('[]').split()
            return float(np.mean([float(v) for v in vals]))
    raise RuntimeError(f"could not parse <n_a> from run_ed.py output:\n{out.stdout[-2000:]}")


print(f"kanamori_phonon mu calibration (exact ED probe): beta={args.beta:g} U={args.U:g} "
      f"J={args.J:g} V={args.V:g} g={args.g} omega_0={args.omega_0:g}")
# The half-filling mu is what model.py computes when --mu is not given; use it as the
# starting guess, which is usually within ~1 of the answer.
sys.path.insert(0, HERE)
import model as model_def  # noqa: E402
guess_args = argparse.Namespace(beta=args.beta, U=args.U, J=args.J, V=args.V,
                                eps_bath=args.eps_bath, omega_0=args.omega_0,
                                g=args.g, mu=None)
mu_half = float(model_def.Model(guess_args).mu[0])
print(f"  half filling is mu = {mu_half:.6f} (particle-hole symmetric, phonon shift included);"
      f" scanning for n = {args.target_n}")

mu, n_achieved, samples = calibrate.bisect_mu(
    density, target_n=args.target_n, mu_guess=mu_half, tol=args.tol, step=0.5, verbose=True)

variable = f"MU_B{args.beta:g}_N{str(args.target_n).replace('.', '')}"
print(calibrate.report(mu, n_achieved, args.target_n,
                       label=f"kanamori_phonon, beta = {args.beta:g}, g = {args.g}",
                       variable=variable))
