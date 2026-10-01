# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""CTHYB run for the Hubbard-Holstein benchmark (model: model.py).

Uniform g makes every D0 vertex Lang-Firsov eligible, so `--lang_firsov False` (all stochastic)
is the internal cross-check. Saves Sigma from G_l and, as Sigma_alt, from G(tau); <N(tau)N(0)>
from O_tau and, as corr_alt, from the Legendre kink estimator Q_tau.
"""
import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.gfs import Fourier
from triqs_cthyb import Solver

import model as M  # puts common/ on sys.path
from common import kernels, selfenergy, str2bool


def add_cthyb_args(parser):
    parser.add_argument("--lang_firsov", type=str2bool, default=True,
                        help="Route the density coupling through Lang-Firsov (False: fully stochastic)")
    parser.add_argument("--dyn_n_l", type=int, default=50, help="Legendre coefficients for the Lang-Firsov kernel")
    parser.add_argument("--n_l", type=int, default=50, help="Legendre coefficients for G_l")
    parser.add_argument("--measure_O_tau_min_ins", type=int, default=50,
                        help="Minimum insertions for the O_tau measurement")
    parser.add_argument("--density_matrix", type=str2bool, default=True,
                        help="Measure the density matrix (and use_norm_as_weight); corr_alt needs it")


args = M.parse_args("CTHYB single-orbital Hubbard-Holstein benchmark", add_cthyb_args)
model = M.Model(args)
if mpi.is_master_node():
    print(model.report())
    if abs(args.filling - 0.5) < 1e-12 and args.mu is None:
        info = model.check_half_filling_mu()
        print(f"  mu = U/2 - g^2/omega_0^2 confirmed: {info['mu']:.8f} vs {info['expected']:.8f}; "
              f"K'(0)={info['kprime_0']:.6f}")

d0 = model.d0(half_prefactor_action=True)

S = Solver(beta=model.beta, gf_struct=M.GF_STRUCT, n_iw=model.n_iw, n_tau=model.n_tau,
           n_l=args.n_l, n_tau_bosonic=model.n_tau_bosonic, delta_interface=True)
S.Delta_tau << Fourier(model.delta_iw())
for (s1, s2), d in d0.items():
    S.D0_tau[s1, s2] << kernels.as_gf(d, model.beta)

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
    mu, delta_block = model.sigma_inputs()

    sigma_l = selfenergy.sigma_from_G_l(S.G_l, model.n_iw, mu, delta_block)
    sigma_tau = selfenergy.sigma_from_G_tau(S.G_tau, model.n_iw, mu, delta_block)
    w_n, sigma_up = selfenergy.positive_frequency_part(sigma_l["up"])
    _, sigma_up_alt = selfenergy.positive_frequency_part(sigma_tau["up"])

    G_up = S.G_tau["up"]
    density = selfenergy.density_from_G_iw(selfenergy.G_iw_from_G_l(S.G_l, model.n_iw))

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
        A["w_n"], A["Sigma"], A["Sigma_alt"] = w_n, sigma_up, sigma_up_alt
        A["tau_corr"], A["corr"] = np.array([float(t) for t in S.O_tau.mesh]), S.O_tau.data.real
        if nn_kink is not None:
            A["corr_alt"] = nn_kink
        A["density"], A["average_sign"] = density, S.average_sign
        A["pert_order"] = S.perturbation_order_total.data
        # Counts stochastic vertices only, so None when all are analytic; the plots read NaN.
        if S.perturbation_order_dyn is not None:
            A["pert_order_dyn"] = S.perturbation_order_dyn.data
        A["G_tau_gf"], A["G_l_gf"] = S.G_tau, S.G_l
    print(f"Saved {path}")
