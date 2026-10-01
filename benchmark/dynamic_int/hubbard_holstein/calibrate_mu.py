# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""Find the mu giving a target density for the Hubbard-Holstein model, with a CTSEG probe, to
pin in run_hubbard_holstein.sh:

    python calibrate_mu.py --beta 10  --target_n 0.75
    python calibrate_mu.py --beta 100 --target_n 0.75
"""
import triqs.utility.mpi as mpi

import model as M  # puts common/ on sys.path
from common import calibrate

args = M.parse_args("Calibrate mu for the Hubbard-Holstein benchmark", calibrate.add_calibration_args)
args.filling, args.mu = 0.5, None
mu_half = M.Model(args).mu

if mpi.is_master_node():
    print(f"hubbard_holstein mu calibration: beta={args.beta:g} U={args.U:g} g={args.g:g} "
          f"omega_0={args.omega_0:g} bath={args.bath}")
    print(f"  half filling is mu = {mu_half:.6f} (exact, = U/2 - g^2/omega_0^2); "
          f"scanning for n = {args.target_n}")

density = calibrate.ctseg_probe(M.Model, args, M.GF_STRUCT, lambda model: (None, model.d0(half_prefactor_action=True)))
mu, n_achieved, samples = calibrate.bisect_mu(
    density, target_n=args.target_n, mu_guess=mu_half, tol=args.tol, verbose=mpi.is_master_node())

if mpi.is_master_node():
    variable = f"MU_B{args.beta:g}_N{str(args.target_n).replace('.', '')}"
    print(calibrate.report(mu, n_achieved, args.target_n,
                           label=f"hubbard_holstein, beta = {args.beta:g}, g = {args.g:g}, "
                                 f"omega_0 = {args.omega_0:g}, bath = {args.bath}",
                           variable=variable))
