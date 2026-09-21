# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""Find the mu giving a target density for the Hubbard-Holstein model, to pin in
run_hubbard_holstein.sh so every solver runs the identical Hamiltonian.

CTSEG is the probe: sign-free for a retarded density-density interaction and much faster
than CTHYB, so the whole scan costs less than one production point.

    module load modules/2.5-beta1 && module load triqs/multiorbital
    python calibrate_mu.py --beta 10  --target_n 0.75
    python calibrate_mu.py --beta 100 --target_n 0.75
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
                        help="MC cycles per probe; the bisection cannot resolve mu below its noise")
    parser.add_argument("--tol", type=float, default=2e-3, help="Tolerance on the density")


args = M.parse_args("Calibrate mu for the Hubbard-Holstein benchmark", add_calibration_args)
args.filling, args.mu = 0.5, None
mu_half = M.Model(args).mu

if mpi.is_master_node():
    print(f"hubbard_holstein mu calibration: beta={args.beta:g} U={args.U:g} g={args.g:g} "
          f"omega_0={args.omega_0:g} bath={args.bath}")
    print(f"  half filling is mu = {mu_half:.6f} (exact, = U/2 - g^2/omega_0^2); "
          f"scanning for n = {args.target_n}")


def density(mu):
    probe = argparse.Namespace(**vars(args))
    probe.mu, probe.filling = mu, args.target_n
    model = M.Model(probe)

    S = Solver(gf_struct=M.GF_STRUCT, beta=model.beta, n_tau=model.n_tau,
               n_tau_bosonic=model.n_tau_bosonic)
    S.Delta_tau << Fourier(model.delta_iw())
    for (s1, s2), d in model.d0(half_prefactor_action=True).items():
        S.D0_tau[s1, s2] << kernels.as_gf(d, model.beta, target_shape=(1, 1))

    S.solve(h_int=model.h_int(), h_loc0=model.h_loc0(),
            length_cycle=args.length_cycle,
            n_warmup_cycles=max(args.probe_cycles // 20, 500),
            n_cycles=args.probe_cycles,
            measure_nn_tau=False, measure_F_tau=False, measure_pert_order=False,
            measure_densities=True)

    # Direct time-average measurement: -G(beta) is far too noisy for a short probe and made
    # an earlier version of this scan non-monotonic.
    return float(np.mean([S.results.densities[bl][i]
                          for bl, size in M.GF_STRUCT for i in range(size)]))


mu, n_achieved, samples = calibrate.bisect_mu(
    density, target_n=args.target_n, mu_guess=mu_half, tol=args.tol, verbose=mpi.is_master_node())

if mpi.is_master_node():
    variable = f"MU_B{args.beta:g}_N{str(args.target_n).replace('.', '')}"
    print(calibrate.report(mu, n_achieved, args.target_n,
                           label=f"hubbard_holstein, beta = {args.beta:g}, g = {args.g:g}, "
                                 f"omega_0 = {args.omega_0:g}, bath = {args.bath}",
                           variable=variable))
