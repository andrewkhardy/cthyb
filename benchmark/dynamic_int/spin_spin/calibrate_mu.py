# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""Find the mu giving a target density for the spin-spin model, with a CTSEG probe, to pin in
run_spin_spin.sh (MU_B10_N075 / MU_B100_N075):

    python calibrate_mu.py --beta 10  --target_n 0.75
    mpirun -n 16 python calibrate_mu.py --beta 100 --target_n 0.75
"""
import triqs.utility.mpi as mpi

import model as M  # puts common/ on sys.path
from common import calibrate

args = M.parse_args("Calibrate mu for the single-orbital spin-spin benchmark", calibrate.add_calibration_args)
# The scan sets mu itself, so the model must not demand one up front.
args.filling = 0.5
args.mu = None
mu_half = M.Model(args).mu

if mpi.is_master_node():
    print(f"spin_spin mu calibration: beta={args.beta:g} U={args.U:g} J={args.J:g} "
          f"jperp={args.jperp:g} szsz={args.szsz:g} bath={args.bath}")
    print(f"  half filling is mu = {mu_half:.6f} (exact); scanning for n = {args.target_n}")
    print(f"  probe: CTSEG, {args.probe_cycles} cycles/rank")

density = calibrate.ctseg_probe(M.Model, args, M.GF_STRUCT,
                                lambda model: model.spin_couplings(half_prefactor_action=True))
mu, n_achieved, samples = calibrate.bisect_mu(
    density, target_n=args.target_n, mu_guess=mu_half,
    tol=args.tol, verbose=mpi.is_master_node())

if mpi.is_master_node():
    variable = f"MU_B{args.beta:g}_N{str(args.target_n).replace('.', '')}"
    print(calibrate.report(mu, n_achieved, args.target_n,
                           label=f"spin_spin, beta = {args.beta:g}, bath = {args.bath}",
                           variable=variable))
