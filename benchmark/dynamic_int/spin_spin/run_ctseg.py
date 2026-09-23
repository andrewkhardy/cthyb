# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""CTSEG run for the single-orbital spin-spin benchmark (model and conventions: model.py).

CTSEG takes exactly the same action as CTHYB -- same Jperp_tau, same D0_tau -- since both
carry the explicit 1/2 (`half_prefactor_action=True`), so any difference between the two is
solver physics.

Two self-energies are saved:

  Sigma       the improved estimator, Sigma = F(iw)/G(iw), from measure_F_tau. Note this
              route uses no G0, mu or Delta whatsoever, so agreeing with the Dyson one is a
              genuine check on the chemical-potential bookkeeping rather than a tautology.
  Sigma_alt   the Dyson inversion, through the same shared code path every other solver uses.

Run under `triqs/multiorbital`, which now carries cthyb, ctseg and ctint together.
"""
import os
import sys

import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.gfs import Fourier
from triqs_ctseg import Solver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402
from common import kernels, selfenergy  # noqa: E402

args = M.parse_args("CTSEG single-orbital spin-spin benchmark")
model = M.Model(args)
if mpi.is_master_node():
    print(model.report())

jperp_tau, d0 = model.spin_couplings(half_prefactor_action=True)

S = Solver(gf_struct=M.GF_STRUCT, beta=model.beta, n_tau=model.n_tau,
           n_tau_bosonic=model.n_tau_bosonic)
S.Delta_tau << Fourier(model.delta_iw())
S.Jperp_tau << kernels.as_gf(jperp_tau, model.beta, target_shape=(1, 1))
for (s1, s2), d in d0.items():
    S.D0_tau[s1, s2] << kernels.as_gf(d, model.beta, target_shape=(1, 1))

S.solve(h_int=model.h_int(), h_loc0=model.h_loc0(),
        length_cycle=args.length_cycle, n_warmup_cycles=args.n_warmup_cycles,
        n_cycles=args.n_cycles, max_time=args.max_time,
        measure_nn_tau=True, measure_F_tau=True, measure_pert_order=True,
        measure_densities=True, **model.seed_kwargs())

if mpi.is_master_node():
    r = S.results
    mu, delta_block = model.sigma_inputs()

    G_iw = selfenergy.G_iw_from_G_tau(r.G_tau, model.n_iw)
    sigma_dyson = selfenergy.sigma_from_G_iw(G_iw, mu, delta_block)
    sigma_improved = selfenergy.sigma_from_F_tau(r.F_tau, G_iw) if r.F_tau is not None else None

    preferred = sigma_improved if sigma_improved is not None else sigma_dyson
    w_n, sigma_up = selfenergy.positive_frequency_part(preferred["up"])
    _, sigma_up_alt = selfenergy.positive_frequency_part(sigma_dyson["up"])
    if sigma_improved is None:
        print("  NOTE: F_tau unavailable; Sigma is the Dyson inversion, same as Sigma_alt.")

    nn = r.nn_tau
    szsz = 0.25 * (nn["up", "up"].data[:, 0, 0] + nn["down", "down"].data[:, 0, 0]
                   - nn["up", "down"].data[:, 0, 0] - nn["down", "up"].data[:, 0, 0]).real

    G_up = r.G_tau["up"]
    # Direct time-average measurement, lower variance than -G(beta); see
    # common/selfenergy.density_from_G_iw for why that matters.
    density = np.array([r.densities[bl][i] for bl, size in M.GF_STRUCT for i in range(size)])
    diag = selfenergy.diagnose(sigma_up, w_n, mu=mu if abs(args.filling - 0.5) < 1e-12 else None)
    print("  " + diag["text"])
    print(f"  average sign = {r.average_sign:.4f}   <n> = {np.round(density, 5)}")

    path = model.output_file("ctseg")
    with HDFArchive(path, "w") as A:
        A["params"] = {k: v for k, v in vars(args).items() if v is not None}
        A["beta"], A["mu"] = model.beta, mu
        A["tau_G"], A["G"] = np.array([float(t) for t in G_up.mesh]), G_up.data[:, 0, 0].real
        A["w_n"], A["Sigma"], A["Sigma_alt"] = w_n, sigma_up, sigma_up_alt
        A["tau_corr"], A["corr"] = np.array([float(t) for t in nn["up", "up"].mesh]), szsz
        A["density"], A["average_sign"] = density, r.average_sign
        if getattr(r, "pert_order_Jperp", None) is not None:
            A["pert_order_dyn"] = np.asarray(r.pert_order_Jperp)
        A["G_tau_gf"], A["nn_tau_gf"] = r.G_tau, nn
    print(f"Saved {path}")
