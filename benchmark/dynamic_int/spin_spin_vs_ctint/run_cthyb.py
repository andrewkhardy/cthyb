# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""CTHYB input for the single-orbital spin-spin benchmark (model and conventions: spin_spin_common.py).

Sz.Sz goes through D0_tau (Lang-Firsov when lang_firsov=True), s+s- through Jperp_tau
(always the stochastic insert_dyn/remove_dyn path)."""
import triqs.utility.mpi as mpi
from triqs.atom_diag import trace_rho_op
from triqs.gfs import Fourier
from triqs_cthyb import Solver

import spin_spin_common as common


def add_cthyb_args(parser):
    parser.add_argument("--lang_firsov", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True,
                        help="Route the Sz.Sz (D0) part through Lang-Firsov (False: stochastic)")
    parser.add_argument("--dyn_n_l", type=int, default=50, help="Legendre coefficients for the Lang-Firsov kernel")
    parser.add_argument("--measure_O_tau_min_ins", type=int, default=50, help="Minimum insertions for the O_tau measurement")


args = common.parse_args("CTHYB single-orbital spin-spin benchmark", add_cthyb_args)
g0, Q_tau = common.load_bath()
jperp_tau, d0 = common.spin_couplings(args.J, Q_tau, args.jperp, args.szsz, half_prefactor_action=True)

S = Solver(beta=common.BETA, gf_struct=common.GF_STRUCT, n_iw=len(g0.mesh) // 2, n_tau=common.N_TAU,
           n_tau_bosonic=common.N_TAU_BOSONIC, delta_interface=True)
S.Delta_tau << Fourier(common.hybridization(g0, args.U))
S.Jperp_tau << jperp_tau
for (s1, s2), d in d0.items():
    S.D0_tau[s1, s2] << d

S.solve(h_int=common.h_int(args.U), h_loc0=common.h_loc0(args.U),
        length_cycle=100, n_warmup_cycles=args.n_warmup_cycles, n_cycles=args.n_cycles,
        measure_pert_order=True, measure_O_tau=(common.SZ, common.SZ),
        measure_O_tau_min_ins=args.measure_O_tau_min_ins,
        measure_D0_corr=True, measure_density_matrix=True, use_norm_as_weight=True,
        lang_firsov=args.lang_firsov, dyn_n_l=args.dyn_n_l)

if mpi.is_master_node():
    G_up = S.G_tau["up"]
    # Kink (Legendre) estimator, as in holstein.py: Q_tau is <n_a(tau) n_b(0)> minus its
    # equal-time value, so add back <Sz^2> from the density matrix.
    Sz2 = trace_rho_op(S.density_matrix, common.SZ * common.SZ, S.h_loc_diagonalization).real
    Q = S.Q_tau
    SzSz_LF = 0.25 * (Q["up", "up"].data[:, 0, 0] + Q["down", "down"].data[:, 0, 0]
                      - Q["up", "down"].data[:, 0, 0] - Q["down", "up"].data[:, 0, 0]).real + Sz2
    common.save_results(common.output_file(args, "cthyb", tag=f"_lf-{args.lang_firsov}"), args, "cthyb",
                        common.tau_points(G_up), G_up.data[:, 0, 0].real,
                        common.tau_points(S.O_tau), S.O_tau.data.real,
                        average_sign=S.average_sign, pert_order_jperp=S.perturbation_order_dyn,
                        raw={"G_tau": S.G_tau, "O_tau": S.O_tau, "K_n": S.K_n, "Q_tau": Q, "Q_l": S.Q_l,
                             "SzSz_offset": Sz2, "tau_SzSz_LF": common.tau_points(Q["up", "up"]), "SzSz_LF": SzSz_LF})
