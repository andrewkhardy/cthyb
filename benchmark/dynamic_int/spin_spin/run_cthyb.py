# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""CTHYB run for the single-orbital spin-spin benchmark (model and conventions: model.py).

Sz.Sz goes in through D0_tau (analytic Lang-Firsov when lang_firsov=True), s+s- through
Jperp_tau, which is always the stochastic insert_dyn/remove_dyn path.

Two independent estimators are saved for each of the two observables that matter, because
the pair bounds the systematic error rather than hiding it:

  Sigma       from the Legendre G_l (preferred: G_l filters the noise before the Dyson
              inversion, which matters at beta = 100 where inverting a noisy G(tau) is
              unusable at high frequency)
  Sigma_alt   from G(tau) by Fourier + Dyson

  corr        <Sz(tau)Sz(0)> from the O_tau insertion measurement
  corr_alt    the same from the Legendre kink estimator (Q_tau)
"""
import os
import sys

import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.gfs import Fourier
from triqs_cthyb import Solver

# Put the benchmark root on the path explicitly rather than relying on model.py having
# been imported first, so reordering these imports cannot break the run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402
from common import kernels, selfenergy  # noqa: E402


def add_cthyb_args(parser):
    parser.add_argument("--lang_firsov", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True,
                        help="Route the Sz.Sz (D0) part through Lang-Firsov (False: fully stochastic)")
    parser.add_argument("--dyn_n_l", type=int, default=50, help="Legendre coefficients for the Lang-Firsov kernel")
    parser.add_argument("--n_l", type=int, default=50, help="Legendre coefficients for G_l")
    parser.add_argument("--measure_O_tau_min_ins", type=int, default=50,
                        help="Minimum insertions for the O_tau measurement")
    parser.add_argument("--move_double", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True,
                        help="Four-operator insert/remove moves. At beta = 100 they accept 0.05%% of "
                             "proposals and take 40%% of move time; not needed for ergodicity here")
    parser.add_argument("--move_dyn_local", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True,
                        help="Let insert_dyn place both ends of a spin-flip vertex in one operator-free "
                             "stretch for part of its proposals (False: all uniform on [0, beta))")
    parser.add_argument("--spin_flip_move", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=False,
                        help="Global up <-> down swap of every operator, Jperp vertices included. A symmetry "
                             "of this model, so it mixes the two moment orientations at no cost")
    parser.add_argument("--density_matrix", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True,
                        help="Measure the density matrix (implies use_norm_as_weight). Needed for the "
                             "equal-time offset that turns Q_tau into the full <n_a(tau)n_b(0)>, so the "
                             "Legendre kink estimator is only saved when this is on. Turn it off for "
                             "local smoke tests: the post-processing needs triqs' atom_diag to be built "
                             "with c2py, and a legacy-cpp2py triqs raises 'Can not wrap AtomDiagReal'")


args = M.parse_args("CTHYB single-orbital spin-spin benchmark", add_cthyb_args)
model = M.Model(args)
if mpi.is_master_node():
    print(model.report())
    # Free consistency check on the sign conventions; only meaningful at half filling.
    if abs(args.filling - 0.5) < 1e-12 and args.mu is None:
        info = model.check_half_filling_mu()
        print(f"  mu = U/2 confirmed: W_shift_offdiag={info['W_shift'][0, 1]:+.6f} "
              f"level_shift={info['level_shift'][0]:+.6f}")

jperp_tau, d0 = model.spin_couplings(half_prefactor_action=True)

spin_flip = {}
if args.spin_flip_move:
    # Accepted every time here (an exact symmetry), but each proposal costs ~0.5 ms: its
    # Lang-Firsov ratio runs over all pairs of the ~90 relabelled operators. At 0.05 that was
    # 28% of the run for 680k flips per chain; 0.002 still gives ~25k, far more than needed.
    spin_flip = dict(move_global={"spin_flip": {("up", 0): ("down", 0), ("down", 0): ("up", 0)}},
                     move_global_full=True, move_global_prob=0.002)

S = Solver(beta=model.beta, gf_struct=M.GF_STRUCT, n_iw=model.n_iw, n_tau=model.n_tau,
           n_l=args.n_l, n_tau_bosonic=model.n_tau_bosonic, delta_interface=True)
S.Delta_tau << Fourier(model.delta_iw())
S.Jperp_tau << kernels.as_gf(jperp_tau, model.beta, target_shape=(1, 1))
for (s1, s2), d in d0.items():
    S.D0_tau[s1, s2] << kernels.as_gf(d, model.beta, target_shape=(1, 1))

S.solve(h_int=model.h_int(), h_loc0=model.h_loc0(),
        length_cycle=args.length_cycle, n_warmup_cycles=args.n_warmup_cycles, move_double=args.move_double,
        move_dyn_local=args.move_dyn_local, **spin_flip,
        n_cycles=args.n_cycles, max_time=args.max_time,
        measure_G_tau=True, measure_G_l=True,
        measure_pert_order=True,
        measure_O_tau=(M.SZ, M.SZ), measure_O_tau_min_ins=args.measure_O_tau_min_ins,
        measure_D0_corr=True,
        measure_density_matrix=args.density_matrix, use_norm_as_weight=args.density_matrix,
        lang_firsov=args.lang_firsov, dyn_n_l=args.dyn_n_l, **model.seed_kwargs())

if mpi.is_master_node():
    mu, delta_block = model.sigma_inputs()

    # Preferred and cross-check Sigma, both built from the *input* mu and Delta.
    sigma_l = selfenergy.sigma_from_G_l(S.G_l, model.n_iw, mu, delta_block)
    sigma_tau = selfenergy.sigma_from_G_tau(S.G_tau, model.n_iw, mu, delta_block)
    w_n, sigma_up = selfenergy.positive_frequency_part(sigma_l["up"])
    _, sigma_up_alt = selfenergy.positive_frequency_part(sigma_tau["up"])

    G_up = S.G_tau["up"]
    tau_G = np.array([float(t) for t in G_up.mesh])
    # With the density matrix measured, <c^dag c> from it directly: an equal-time average
    # rather than a tail extrapolation of G_l, so far lower variance. Otherwise from the
    # Legendre G(iw) (see common/selfenergy.density_from_G_iw).
    if args.density_matrix:
        density = np.array([S.orbital_occupations[bl][i, i].real for bl, size in M.GF_STRUCT for i in range(size)])
    else:
        density = selfenergy.density_from_G_iw(selfenergy.G_iw_from_G_l(S.G_l, model.n_iw))

    # Legendre kink estimator. With measure_density_matrix=True the Python Solver has
    # already added the equal-time <n_a n_b> to Q_tau, so Q_tau is the full
    # <n_a(tau) n_b(0)> -- do not add an offset again (solver.py since 93b222f). Without
    # it, Q_tau is only the connected part, which is not the same observable as O_tau, so
    # it is left out rather than saved under a label that would invite comparison.
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
        A["pert_order"], A["pert_order_dyn"] = S.perturbation_order_total.data, S.perturbation_order_dyn.data
        A["G_tau_gf"], A["G_l_gf"] = S.G_tau, S.G_l
    print(f"Saved {path}")
