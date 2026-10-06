# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""CTHYB run for the spin-spin benchmark (model and conventions: model.py).

The Sz.Sz part couples conserved densities and is resummed analytically (Lang-Firsov) when
lang_firsov=True; the spin flips are always sampled stochastically. Saves Sigma from G_l and, as Sigma_alt, from G(tau);
<Sz(tau)Sz(0)> from O_tau and, as corr_alt, from the Legendre kink estimator Q_tau.
"""
import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.gfs import Fourier
from triqs_cthyb import Solver

import model as M  # puts common/ on sys.path
from common import selfenergy, str2bool


def add_cthyb_args(parser):
    parser.add_argument("--lang_firsov", type=str2bool, default=True,
                        help="Route the Sz.Sz (D0) part through Lang-Firsov (False: fully stochastic)")
    parser.add_argument("--dyn_n_l", type=int, default=50, help="Legendre coefficients for the Lang-Firsov kernel")
    parser.add_argument("--n_l", type=int, default=50, help="Legendre coefficients for G_l")
    parser.add_argument("--move_double", type=str2bool, default=True, help="Four-operator insert/remove moves")
    parser.add_argument("--spin_flip_move", type=str2bool, default=False,
                        help="Global up <-> down swap of every operator, a symmetry of this model")
    parser.add_argument("--density_matrix", type=str2bool, default=True,
                        help="Measure the density matrix (and use_norm_as_weight); corr_alt needs it")


args = M.parse_args("CTHYB single-orbital spin-spin benchmark", add_cthyb_args)
model = M.Model(args)
if mpi.is_master_node():
    print(model.report())

spin_flip = {}
if args.spin_flip_move:
    # Always accepted, but each proposal re-evaluates the Lang-Firsov ratio of every operator.
    spin_flip = dict(move_global={"spin_flip": {("up", 0): ("down", 0), ("down", 0): ("up", 0)}},
                     move_global_full=True, move_global_prob=0.002)

S = Solver(beta=model.beta, gf_struct=M.GF_STRUCT, n_iw=model.n_iw, n_tau=model.n_tau,
           n_l=args.n_l, n_tau_bosonic=model.n_tau_bosonic, delta_interface=True)
S.Delta_tau << Fourier(model.delta_iw())

# S_dyn = 1/2 int int lambda(tau - tau') [szsz Sz Sz + jperp (S+ S- + S- S+) / 2]
lam = model.spin_kernel
S.add_dyn_int(args.szsz * lam, M.SZ, M.SZ)
S.add_dyn_int(args.jperp * lam / 2, M.SP, M.SM)
S.add_dyn_int(args.jperp * lam / 2, M.SM, M.SP)

S.solve(h_int=model.h_int(), h_loc0=model.h_loc0(),
        length_cycle=args.length_cycle, n_warmup_cycles=args.n_warmup_cycles, move_double=args.move_double,
        **spin_flip,
        n_cycles=args.n_cycles, max_time=args.max_time,
        measure_G_tau=True, measure_G_l=True,
        measure_pert_order=True,
        measure_O_tau=(M.SZ, M.SZ),
        measure_D0_corr=True,
        measure_density_matrix=args.density_matrix, use_norm_as_weight=args.density_matrix,
        lang_firsov=args.lang_firsov, dyn_n_l=args.dyn_n_l, **model.seed_kwargs())

if mpi.is_master_node():
    mu, delta_block = model.sigma_inputs()

    sigma_l = selfenergy.sigma_from_G_l(S.G_l, model.n_iw, mu, delta_block)
    sigma_tau = selfenergy.sigma_from_G_tau(S.G_tau, model.n_iw, mu, delta_block)
    w_n, sigma_up = selfenergy.positive_frequency_part(sigma_l["up"])
    _, sigma_up_alt = selfenergy.positive_frequency_part(sigma_tau["up"])

    G_up = S.G_tau["up"]
    tau_G = np.array([float(t) for t in G_up.mesh])
    # The density matrix gives the equal-time occupations directly, with far less variance
    # than the tail of G_l.
    if args.density_matrix:
        density = np.array([S.orbital_occupations[bl][i, i].real for bl, size in M.GF_STRUCT for i in range(size)])
    else:
        density = selfenergy.density_from_G_iw(selfenergy.G_iw_from_G_l(S.G_l, model.n_iw))

    # With the density matrix measured, the Python Solver has already added the equal-time
    # <n_a n_b> to Q_tau; without it Q_tau is a different observable from O_tau, so not saved.
    Q = S.Q_tau
    szsz_kink = None
    if args.density_matrix and Q is not None:
        szsz_kink = 0.25 * (Q["up", "up"].data[:, 0, 0] + Q["down", "down"].data[:, 0, 0]
                            - Q["up", "down"].data[:, 0, 0] - Q["down", "up"].data[:, 0, 0]).real
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
        A["tau_G"], A["G"] = tau_G, G_up.data[:, 0, 0].real
        A["w_n"], A["Sigma"], A["Sigma_alt"] = w_n, sigma_up, sigma_up_alt
        A["tau_corr"], A["corr"] = np.array([float(t) for t in S.O_tau.mesh]), S.O_tau.data.real
        if szsz_kink is not None:
            A["corr_alt"] = szsz_kink
        A["density"], A["average_sign"] = density, S.average_sign
        A["pert_order"] = S.perturbation_order_total.data
        # Counts stochastic vertices only, so None when all are analytic; the plots read NaN.
        if S.perturbation_order_dyn is not None:
            A["pert_order_dyn"] = S.perturbation_order_dyn.data
        A["G_tau_gf"], A["G_l_gf"] = S.G_tau, S.G_l
    print(f"Saved {path}")
