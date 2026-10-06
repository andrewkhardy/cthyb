# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""CTHYB run of the Hubbard-Holstein model in model.py.

    mpirun -n <N> python run_cthyb.py --beta 10 --filling 0.5 --lang_firsov True

lang_firsov=False samples the phonon vertices stochastically instead of analytically, which
is the internal cross-check. Sigma is saved from G_l and from G_tau, <N(tau) N(0)> from the
O_tau insertions and from Q_tau.
"""
import os
import sys

import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.gfs import Fourier
from triqs_cthyb import Solver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402
from common import selfenergy  # noqa: E402

str_to_bool = lambda x: str(x).lower() in ("true", "1", "yes")


def add_cthyb_args(parser):
    parser.add_argument("--lang_firsov", type=str_to_bool, default=True, help="False samples every vertex stochastically")
    parser.add_argument("--dyn_n_l", type=int, default=50, help="Legendre coefficients for the dynamical interaction")
    parser.add_argument("--n_l", type=int, default=50, help="Legendre coefficients for G_l")
    parser.add_argument("--measure_O_tau_min_ins", type=int, default=50)
    parser.add_argument("--density_matrix", type=str_to_bool, default=True,
                        help="Measure the density matrix (and use_norm_as_weight); needed for the Q_tau estimator")


args = M.parse_args("CTHYB: single-orbital Hubbard-Holstein", add_cthyb_args)
model = M.Model(args)
if mpi.is_master_node():
    print(model.report())

S = Solver(beta=model.beta, gf_struct=M.GF_STRUCT, n_iw=model.n_iw, n_tau=model.n_tau,
           n_l=args.n_l, n_tau_bosonic=model.n_tau_bosonic, delta_interface=True)
S.Delta_tau << Fourier(model.delta_iw())

# Phonon: S_dyn = 1/2 int int g^2 Q(tau - tau') N(tau) N(tau'), N = n_up + n_down
S.add_dyn_int(model.g ** 2 * model.Q, M.N_TOT, M.N_TOT)

S.solve(h_int=model.h_int(), h_loc0=model.h_loc0(),
        length_cycle=args.length_cycle, n_warmup_cycles=args.n_warmup_cycles,
        n_cycles=args.n_cycles, max_time=args.max_time,
        measure_G_tau=True, measure_G_l=True,
        measure_pert_order=True,
        measure_O_tau=(M.N_TOT, M.N_TOT), measure_O_tau_min_ins=args.measure_O_tau_min_ins,
        measure_D0_corr=True,
        measure_density_matrix=args.density_matrix, use_norm_as_weight=args.density_matrix,
        lang_firsov=args.lang_firsov, dyn_n_l=args.dyn_n_l)

if mpi.is_master_node():
    # Sigma from the input mu and Delta, not the solver's h_loc, which Lang-Firsov shifts
    mu, delta_block = model.sigma_inputs()
    sigma_l = selfenergy.sigma_from_G_l(S.G_l, model.n_iw, mu, delta_block)
    sigma_tau = selfenergy.sigma_from_G_tau(S.G_tau, model.n_iw, mu, delta_block)
    w_n, sigma_up = selfenergy.positive_frequency_part(sigma_l["up"])
    _, sigma_up_alt = selfenergy.positive_frequency_part(sigma_tau["up"])

    G_up = S.G_tau["up"]
    density = selfenergy.density_from_G_iw(selfenergy.G_iw_from_G_l(S.G_l, model.n_iw))

    # <N(tau) N(0)> = sum_ss' <n_s(tau) n_s'(0)>; the solver has already added the equal-time part
    nn_kink = None
    if args.density_matrix and S.Q_tau is not None:
        nn_kink = sum(S.Q_tau[s1, s2].data[:, 0, 0].real for s1 in M.SPINS for s2 in M.SPINS)
    else:
        print("  NOTE: no density matrix, so the Q_tau estimator (corr_alt) is not saved.")

    diag = selfenergy.diagnose(sigma_up, w_n, mu=mu if abs(args.filling - 0.5) < 1e-12 else None)
    print("  " + diag["text"])
    print(f"  average sign = {S.average_sign:.4f}   <n> = {np.round(density, 5)}")

    path = model.output_file("cthyb", tag=f"lf-{args.lang_firsov}")
    with HDFArchive(path, "w") as A:
        A["params"] = {k: v for k, v in vars(args).items() if v is not None}
        A["beta"], A["mu"] = model.beta, mu
        A["tau_G"], A["G"] = np.array([float(t) for t in G_up.mesh]), G_up.data[:, 0, 0].real
        A["w_n"], A["Sigma"], A["Sigma_alt"] = w_n, sigma_up, sigma_up_alt
        A["tau_corr"], A["corr"] = np.array([float(t) for t in S.O_tau.mesh]), S.O_tau.data.real
        if nn_kink is not None:
            A["corr_alt"] = nn_kink
        A["density"], A["average_sign"] = density, S.average_sign
        A["pert_order"] = S.perturbation_order_total.data
        # Only stochastic vertices are counted, so this is None with lang_firsov=True
        if S.perturbation_order_dyn is not None:
            A["pert_order_dyn"] = S.perturbation_order_dyn.data
        A["G_tau_gf"], A["G_l_gf"] = S.G_tau, S.G_l
    print(f"Saved {path}")
