# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""Find the mu that gives a target density for the single-orbital spin-spin model, so it
can be pinned in run_spin_spin.sh and shared by every solver.

Uses CTSEG as the probe: sign-free for this model and ~14x faster than CTHYB, so a whole
scan costs less than one production point. Any accurate solver would give the same n(mu) --
the point of calibrating once is that all three then run the *same* Hamiltonian.

    module load modules/2.5-beta1 && module load triqs/multiorbital
    python calibrate_mu.py --beta 10  --target_n 0.75
    mpirun -n 16 python calibrate_mu.py --beta 100 --target_n 0.75

Paste the printed value into MU_B10_N075 / MU_B100_N075.
"""
import argparse
import os
import sys

import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import Fourier
from triqs_ctseg import Solver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402
from common import calibrate, kernels  # noqa: E402


def add_calibration_args(parser):
    parser.add_argument("--target_n", type=float, default=0.75,
                        help="Target density per spin-orbital (0.5 is half filling)")
    parser.add_argument("--probe_cycles", type=int, default=20000,
                        help="MC cycles per probe. Keep small: the bisection cannot resolve mu "
                             "below the probe's own noise, so precision here is wasted")
    parser.add_argument("--tol", type=float, default=2e-3,
                        help="Tolerance on the density, matched to the probe's statistical error")


args = M.parse_args("Calibrate mu for the single-orbital spin-spin benchmark", add_calibration_args)
# The scan sets mu itself, so the model must not demand one up front.
args.filling = 0.5
args.mu = None
base = M.Model(args)
mu_half = base.mu

if mpi.is_master_node():
    print(f"spin_spin mu calibration: beta={args.beta:g} U={args.U:g} J={args.J:g} "
          f"jperp={args.jperp:g} szsz={args.szsz:g} bath={args.bath}")
    print(f"  half filling is mu = {mu_half:.6f} (exact); scanning for n = {args.target_n}")
    print(f"  probe: CTSEG, {args.probe_cycles} cycles/rank")


def density(mu):
    """Mean density per spin-orbital from a short CTSEG run at this mu."""
    probe_args = argparse.Namespace(**vars(args))
    probe_args.mu = mu
    probe_args.filling = args.target_n
    model = M.Model(probe_args)

    jperp_tau, d0 = model.spin_couplings(half_prefactor_action=True)
    S = Solver(gf_struct=M.GF_STRUCT, beta=model.beta, n_tau=model.n_tau,
               n_tau_bosonic=model.n_tau_bosonic)
    S.Delta_tau << Fourier(model.delta_iw())
    S.Jperp_tau << kernels.as_gf(jperp_tau, model.beta, target_shape=(1, 1))
    for (s1, s2), d in d0.items():
        S.D0_tau[s1, s2] << kernels.as_gf(d, model.beta, target_shape=(1, 1))

    S.solve(h_int=model.h_int(), h_loc0=model.h_loc0(),
            length_cycle=args.length_cycle,
            n_warmup_cycles=max(args.probe_cycles // 20, 500),
            n_cycles=args.probe_cycles,
            measure_nn_tau=False, measure_F_tau=False, measure_pert_order=False,
            measure_densities=True)

    # CTSEG's direct density measurement: a plain time average, far lower variance than
    # reading -G(beta). With -G(beta) a 20k-cycle probe scattered by +-0.15 and made the
    # measured n(mu) non-monotonic, which broke the bisection.
    n_per_spin = [S.results.densities[bl][i] for bl, size in M.GF_STRUCT for i in range(size)]
    return float(np.mean(n_per_spin))


mu, n_achieved, samples = calibrate.bisect_mu(
    density, target_n=args.target_n, mu_guess=mu_half,
    tol=args.tol, verbose=mpi.is_master_node())

if mpi.is_master_node():
    # Use every probe, not just the final bisection step -- the probe is stochastic.
    variable = f"MU_B{args.beta:g}_N{str(args.target_n).replace('.', '')}"
    print(calibrate.report(mu, n_achieved, args.target_n,
                           label=f"spin_spin, beta = {args.beta:g}, bath = {args.bath}",
                           variable=variable))
