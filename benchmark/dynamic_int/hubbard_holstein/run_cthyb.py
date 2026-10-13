# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""CTHYB run for the Hubbard-Holstein benchmark (model: model.py).

The phonon couples to the total density N, which is conserved, so every vertex is resummed
analytically; `--lang_firsov False` (all stochastic) is the internal cross-check. The solver runs in
its DLR mode: Delta goes in, and G comes out, on the DLR Matsubara mesh (common/grids.py). Saves
Sigma on the positive DLR frequencies; <N(tau)N(0)> from O_tau and, as corr_alt, from the Legendre
kink estimator Q_tau.
"""
import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs_cthyb import Solver

import model as M  # puts common/ on sys.path
from common import grids, selfenergy, str2bool


def add_cthyb_args(parser):
    parser.add_argument("--lang_firsov", type=str2bool, default=True,
                        help="Route the density coupling through Lang-Firsov (False: fully stochastic)")
    parser.add_argument("--dyn_n_l", type=int, default=50, help="Legendre coefficients for the Lang-Firsov kernel")
    grids.add_dlr_args(parser)
    parser.add_argument("--density_matrix", type=str2bool, default=True,
                        help="Measure the density matrix (and use_norm_as_weight); corr_alt needs it")


args = M.parse_args("CTHYB single-orbital Hubbard-Holstein benchmark", add_cthyb_args)
model = M.Model(args)
if mpi.is_master_node():
    print(model.report())

S = Solver(beta=model.beta, gf_struct=M.GF_STRUCT, n_iw=model.n_iw, n_tau=model.n_tau,
           n_tau_bosonic=model.n_tau_bosonic, delta_interface=True, dlr_wmax=args.dlr_wmax, dlr_eps=args.dlr_eps)
mu, delta_block = model.sigma_inputs(S.dlr_mesh)
S.Delta_iw << delta_block

# Phonon: S_dyn = 1/2 int int g^2 Q(tau - tau') N(tau) N(tau'), N = n_up + n_down
S.add_dyn_int(model.g ** 2 * model.Q, M.N_TOT, M.N_TOT)

S.solve(h_int=model.h_int(), h_loc0=model.h_loc0(),
        length_cycle=args.length_cycle, n_warmup_cycles=args.n_warmup_cycles,
        n_cycles=args.n_cycles, max_time=args.max_time,
        measure_G_tau=True,
        measure_pert_order=True,
        measure_O_tau=(M.N_TOT, M.N_TOT),
        measure_D0_corr=True,
        measure_density_matrix=args.density_matrix, use_norm_as_weight=args.density_matrix,
        lang_firsov=args.lang_firsov, dyn_n_l=args.dyn_n_l)

if mpi.is_master_node():
    sigma = selfenergy.sigma_from_G_iw(S.G_iw, mu, delta_block)
    w_n, sigma_up = selfenergy.positive_frequency_part(sigma["up"])

    G_up = S.G_tau["up"]
    # The density matrix gives the equal-time occupations directly, with far less variance
    # than the ends of the DLR fit of G(tau).
    if args.density_matrix:
        density = np.array([S.orbital_occupations[bl][i, i].real for bl, size in M.GF_STRUCT for i in range(size)])
    else:
        density = selfenergy.density_from_G_dlr(S.G_dlr)

    # <N(tau)N(0)> summed over ordered spin pairs. With the density matrix measured, the Python
    # Solver has already added the equal-time part to Q_tau; without it Q_tau lacks it.
    Q = S.Q_tau
    nn_kink = None
    if args.density_matrix and Q is not None:
        nn_kink = sum(Q[s1, s2].data[:, 0, 0].real for s1 in M.SPINS for s2 in M.SPINS)
    else:
        print("  NOTE: --density_matrix False, so Q_tau lacks its equal-time offset; "
              "the Legendre kink estimator is not saved.")

    diag = selfenergy.diagnose(sigma_up, w_n, mu=mu if abs(args.filling - 0.5) < 1e-12 else None)
    print("  " + diag["text"])
    print(f"  average sign = {S.average_sign:.4f}   <n> = {np.round(density, 5)}")

    path = model.output_file("cthyb", tag=f"lf-{args.lang_firsov}")
    with HDFArchive(path, "w") as A:
        A["params"] = {k: v for k, v in vars(args).items() if v is not None}
        A["beta"], A["mu"] = model.beta, mu
        A["tau_G"], A["G"] = np.array([float(t) for t in G_up.mesh]), G_up.data[:, 0, 0].real
        A["w_n"], A["Sigma"] = w_n, sigma_up
        A["tau_corr"], A["corr"] = np.array([float(t) for t in S.O_tau.mesh]), S.O_tau.data.real
        if nn_kink is not None:
            A["corr_alt"] = nn_kink
        A["density"], A["average_sign"] = density, S.average_sign
        A["pert_order"] = S.perturbation_order_total.data
        # Counts stochastic vertices only, so None when all are analytic; the plots read NaN.
        if S.perturbation_order_dyn is not None:
            A["pert_order_dyn"] = S.perturbation_order_dyn.data
        A["G_tau_gf"], A["G_dlr"] = S.G_tau, S.G_dlr
    print(f"Saved {path}")
