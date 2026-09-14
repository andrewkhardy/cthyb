# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""CTSEG input for the single-orbital spin-spin benchmark (model and conventions: spin_spin_common.py)."""
import triqs.utility.mpi as mpi
from triqs.gfs import Fourier
from triqs_ctseg import Solver

import spin_spin_common as common

args = common.parse_args("CTSEG single-orbital spin-spin benchmark")
g0, Q_tau = common.load_bath()
jperp_tau, d0 = common.spin_couplings(args.J, Q_tau, args.jperp, args.szsz, half_prefactor_action=True)

S = Solver(gf_struct=common.GF_STRUCT, beta=common.BETA, n_tau=common.N_TAU, n_tau_bosonic=common.N_TAU_BOSONIC)
S.Delta_tau << Fourier(common.hybridization(g0, args.U))
S.Jperp_tau << jperp_tau
for (s1, s2), d in d0.items():
    S.D0_tau[s1, s2] << d

S.solve(h_int=common.h_int(args.U), h_loc0=common.h_loc0(args.U),
        length_cycle=100, n_warmup_cycles=args.n_warmup_cycles, n_cycles=args.n_cycles,
        measure_nn_tau=True, measure_pert_order=True)

if mpi.is_master_node():
    r = S.results
    nn = r.nn_tau
    SzSz = 0.25 * (nn["up", "up"].data[:, 0, 0] + nn["down", "down"].data[:, 0, 0]
                   - nn["up", "down"].data[:, 0, 0] - nn["down", "up"].data[:, 0, 0]).real
    G_up = r.G_tau["up"]
    common.save_results(common.output_file(args, "ctseg"), args, "ctseg",
                        common.tau_points(G_up), G_up.data[:, 0, 0].real,
                        common.tau_points(nn["up", "up"]), SzSz,
                        average_sign=r.average_sign, pert_order_jperp=getattr(r, "pert_order_Jperp", None),
                        raw={"G_tau": r.G_tau, "nn_tau": nn})
