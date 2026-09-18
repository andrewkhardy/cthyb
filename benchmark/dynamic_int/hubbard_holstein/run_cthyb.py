# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""CTHYB run for the single-orbital Hubbard-Holstein benchmark (model: model.py).

The phonon couples to the total charge, so every D0_tau entry -- including the diagonal
(s, s) self-terms -- carries g^2 Q(tau). That uniformity is what makes the coupling
block-constant on the conserved densities, so with `lang_firsov=True` every vertex should
be routed through the analytic Lang-Firsov path and the sign should stay at 1.0. Running
with `lang_firsov=False` forces the same physics through the stochastic expansion, which is
the independent internal cross-check; the two must agree.

Saves two estimators of each key quantity, as in the spin-spin benchmark: Sigma from the
Legendre G_l (preferred) and from G(tau) by Dyson, and <N(tau)N(0)> from the O_tau insertion
measurement and from the Legendre kink estimator.
"""
import os
import sys

import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import Fourier
from triqs_cthyb import Solver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402
from common import io, kernels, selfenergy  # noqa: E402


def add_cthyb_args(parser):
    parser.add_argument("--lang_firsov", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True,
                        help="Route the density coupling through Lang-Firsov (False: fully stochastic)")
    parser.add_argument("--dyn_n_l", type=int, default=50, help="Legendre coefficients for the Lang-Firsov kernel")
    parser.add_argument("--n_l", type=int, default=50, help="Legendre coefficients for G_l")
    parser.add_argument("--measure_O_tau_min_ins", type=int, default=50,
                        help="Minimum insertions for the O_tau measurement")
    parser.add_argument("--density_matrix", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True,
                        help="Measure the density matrix (implies use_norm_as_weight). Needed for the "
                             "equal-time offset on Q_tau, so the Legendre kink estimator is only saved "
                             "when this is on. Turn it off for local smoke tests if triqs' atom_diag "
                             "and cthyb were built against different c2py versions")


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
    S.D0_tau[s1, s2] << kernels.as_gf(d, model.beta, target_shape=(1, 1))

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

    # <N(tau)N(0)> = sum over all ordered spin pairs of <n_s(tau) n_s'(0)>. With
    # measure_density_matrix=True the Python Solver has already added the equal-time part,
    # so Q_tau is the full correlator -- do not add an offset again.
    Q = S.Q_tau
    nn_kink, tau_kink = None, None
    if args.density_matrix and Q is not None:
        nn_kink = sum(Q[s1, s2].data[:, 0, 0].real for s1 in M.SPINS for s2 in M.SPINS)
        tau_kink = np.array([float(t) for t in Q["up", "up"].mesh])
    else:
        print("  NOTE: --density_matrix False, so Q_tau lacks its equal-time offset; "
              "the Legendre kink estimator is not saved.")

    diag = selfenergy.diagnose(sigma_up, w_n, mu=mu if abs(args.filling - 0.5) < 1e-12 else None)
    print("  " + diag["text"])
    print(f"  average sign = {S.average_sign:.4f}   <n> = {np.round(density, 5)}")

    io.save(model.output_file("cthyb", tag=f"lf-{args.lang_firsov}"),
            solver="cthyb", params=vars(args), beta=model.beta, mu=mu,
            tau_G=np.array([float(t) for t in G_up.mesh]), G=G_up.data[:, 0, 0].real,
            w_n=w_n, Sigma=sigma_up, Sigma_alt=sigma_up_alt,
            tau_corr=np.array([float(t) for t in S.O_tau.mesh]), corr=S.O_tau.data.real,
            corr_label=r"$\langle N(\tau)N(0)\rangle$ (O_tau)",
            corr_alt=nn_kink,
            corr_alt_label=r"$\langle N(\tau)N(0)\rangle$ (Legendre kink)" if nn_kink is not None else None,
            density=density, average_sign=S.average_sign,
            pert_order=S.perturbation_order_total, pert_order_dyn=S.perturbation_order_dyn,
            raw={"G_tau": S.G_tau, "G_l": S.G_l, "O_tau": S.O_tau, "Q_tau": Q, "Q_l": S.Q_l,
                 "K_n": S.K_n, "Sigma_iw_legendre": sigma_l, "Sigma_iw_from_tau": sigma_tau,
                 "tau_corr_alt": tau_kink})
