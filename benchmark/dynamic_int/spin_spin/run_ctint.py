# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""CTINT reference for the spin-spin benchmark (model and conventions: model.py).

CTINT's action has no 1/2, so its Jperp and D0 are half of CTSEG's and CTHYB's
(`spin_couplings(half_prefactor_action=False)`). It takes G0_iw instead of Delta_tau: the
G0^-1 = iw + mu - Delta of common/selfenergy.py.
"""
import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.gfs import BlockGf
from triqs_ctint import Solver

import model as M  # puts common/ on sys.path
from common import baths, ctint, selfenergy


def add_ctint_args(parser):
    parser.add_argument("--dlr_wmax", type=float, default=10.0, help="DLR frequency cutoff")
    parser.add_argument("--dlr_eps", type=float, default=1e-10, help="DLR accuracy")
    ctint.add_alpha_args(parser)


args = M.parse_args("CTINT single-orbital spin-spin benchmark (reference)", add_ctint_args)
model = M.Model(args)
if mpi.is_master_node():
    print(model.report())

jperp_tau, d0 = model.spin_couplings(half_prefactor_action=False)
use_jperp, use_szsz = args.jperp != 0, args.szsz != 0

S = Solver(beta=model.beta, gf_struct=M.GF_STRUCT, n_tau=model.n_tau_bosonic,
           use_Jperp=use_jperp, use_D=use_szsz, dlr_wmax=args.dlr_wmax)

mesh = baths.imfreq_mesh(model.beta, model.n_iw)
ctint.set_g0(S, mesh, model.mu, model.delta_iw(), model.n_tau, args.dlr_wmax, args.dlr_eps)

if use_jperp:
    S.Jperp_iw.data[:] = ctint.dlr_imfreq_from_tau(jperp_tau, model.beta, args.dlr_wmax, args.dlr_eps).data[:]
if use_szsz:
    for (s1, s2), d in d0.items():
        S.D0_iw[s1, s2].data[:] = ctint.dlr_imfreq_from_tau(d, model.beta, args.dlr_wmax, args.dlr_eps).data[:]

# D0 = +-spin_kernel/8 has opposite signs in the same- and opposite-spin channels;
# channels in the library's (block1, block2) order.
names = [bl for bl, _ in M.GF_STRUCT]
d0_channels = [d0[bl1, bl2] for bl1 in names for bl2 in names] if use_szsz else []
alpha_kwargs, alpha_report = ctint.alpha_kwargs(args, model.h_int(), d0_channels)
if mpi.is_master_node():
    print(alpha_report)

S.solve(h_int=model.h_int(), **alpha_kwargs,
        length_cycle=args.length_cycle,
        n_warmup_cycles=args.n_warmup_cycles, n_cycles=args.n_cycles, max_time=args.max_time,
        measure_M_iw=True, measure_M_tau=False,
        # On the shared grid rather than the library's 201 points (common/grids.py).
        measure_chiAB_tau=True, chi_A_vec=[M.SZ], chi_B_vec=[M.SZ], n_tau_chi2=model.n_tau,
        post_process=True)

if mpi.is_master_node():
    mu, delta_block = model.sigma_inputs()

    G_tau = {bl: ctint.to_uniform_tau(S.G_iw[bl], model.n_tau) for bl in names}
    G_iw_reg = BlockGf(name_list=names, block_list=[ctint.to_regular_imfreq(S.G_iw[bl], mesh) for bl in names])

    sigma_dyson = selfenergy.sigma_from_G_iw(G_iw_reg, mu, delta_block)
    w_n, sigma_up = selfenergy.positive_frequency_part(sigma_dyson["up"])

    # CTINT's own Sigma (Hartree-Fock + M-matrix post-processing) as the second route.
    sigma_up_alt = None
    if getattr(S, "Sigma_iw", None) is not None:
        _, sigma_up_alt = selfenergy.positive_frequency_part(ctint.to_regular_imfreq(S.Sigma_iw["up"], mesh))

    density = selfenergy.density_from_G_iw(G_iw_reg)
    sz_mean = 0.5 * (density[M.SPIN_INDEX["up"]] - density[M.SPIN_INDEX["down"]])
    chi_tau, szsz = ctint.chi_on_uniform_tau(S.chiAB_tau, sz_mean ** 2, model.n_tau_bosonic)

    diag = selfenergy.diagnose(sigma_up, w_n, mu=mu if abs(args.filling - 0.5) < 1e-12 else None)
    print("  " + diag["text"])
    print(f"  average sign = {S.average_sign:.4f}   average order = {S.average_k:.2f}   "
          f"<n> = {np.round(density, 5)}")

    path = model.output_file("ctint")
    with HDFArchive(path, "w") as A:
        A["params"] = {k: v for k, v in vars(args).items() if v is not None}
        A["beta"], A["mu"] = model.beta, mu
        A["tau_G"], A["G"] = np.array([float(t) for t in G_tau["up"].mesh]), G_tau["up"].data[:, 0, 0].real
        A["w_n"], A["Sigma"] = w_n, sigma_up
        if sigma_up_alt is not None:
            A["Sigma_alt"] = sigma_up_alt
        A["tau_corr"], A["corr"] = np.array([float(t) for t in chi_tau.mesh]), szsz
        A["density"], A["average_sign"] = density, float(np.real(S.average_sign))
        A["G_iw_gf"], A["chiAB_tau_gf"] = S.G_iw, S.chiAB_tau
    print(f"Saved {path}")
