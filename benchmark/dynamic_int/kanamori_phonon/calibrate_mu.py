# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""Find the mu giving a target density for the Kanamori+phonon model, to pin in
run_kanamori_phonon.sh (MU_B10_N075 / MU_B100_N075). The probe is the exact ED of run_ed.py:

    python calibrate_mu.py --beta 10  --target_n 0.75
    python calibrate_mu.py --beta 100 --target_n 0.75
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import calibrate  # noqa: E402
import model as model_def  # noqa: E402

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
    parser.add_argument('--target_n', type=float, default=0.75, help='Target density per spin-orbital')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance on the density')


parser = argparse.ArgumentParser(description='Calibrate mu for the Kanamori+phonon benchmark')
add_args(parser)
args = parser.parse_args()

BASE = ['--beta', str(args.beta), '--U', str(args.U), '--J', str(args.J), '--V', str(args.V),
        '--eps_bath', str(args.eps_bath), '--omega_0', str(args.omega_0),
        '--g', str(args.g[0]), str(args.g[1]), '--n_ph', str(args.n_ph)]

print(f"kanamori_phonon mu calibration (exact ED probe): beta={args.beta:g} U={args.U:g} "
      f"J={args.J:g} V={args.V:g} g={args.g} omega_0={args.omega_0:g}")
guess_args = argparse.Namespace(beta=args.beta, U=args.U, J=args.J, V=args.V,
                                eps_bath=args.eps_bath, omega_0=args.omega_0,
                                g=args.g, mu=None)
mu_half = float(model_def.Model(guess_args).mu[0])
print(f"  half filling is mu = {mu_half:.6f} (particle-hole symmetric, phonon shift included);"
      f" scanning for n = {args.target_n}")

density = calibrate.ed_probe(os.path.join(HERE, 'run_ed.py'), BASE)
mu, n_achieved, samples = calibrate.bisect_mu(
    density, target_n=args.target_n, mu_guess=mu_half, tol=args.tol, step=0.5, verbose=True)

variable = f"MU_B{args.beta:g}_N{str(args.target_n).replace('.', '')}"
print(calibrate.report(mu, n_achieved, args.target_n,
                       label=f"kanamori_phonon, beta = {args.beta:g}, g = {args.g}",
                       variable=variable))
