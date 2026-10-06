# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# CTINT reference run of the spin-spin model in model.py (Jperp and D0 in CTINT's convention,
# half of CTSEG's).
#
#   mpirun -n <N> python run_ctint.py --beta 10 --filling 0.5 --jperp 1 --szsz 1
#
# CTINT takes G0_iw instead of Delta: it gets the G0^-1 = iw + mu - Delta that
# common/selfenergy.py also uses for Sigma.

import os
import sys

import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.gfs import (BlockGf, Gf, MeshDLRImFreq, MeshDLRImTime, fit_gf_dlr, inverse, make_gf_dlr,
                       make_gf_dlr_imfreq, make_gf_from_fourier, make_gf_imtime)
from triqs_ctint import Solver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402
from common import baths, ctint, kernels, selfenergy  # noqa: E402

DLR_WMAX = 10.0
DLR_EPS = 1e-10


def add_ctint_args(parser):
    parser.add_argument("--dlr_wmax", type=float, default=DLR_WMAX, help="DLR frequency cutoff")
    parser.add_argument("--dlr_eps", type=float, default=DLR_EPS, help="DLR accuracy")
    ctint.add_alpha_args(parser)


args = M.parse_args("CTINT single-orbital spin-spin benchmark (reference)", add_ctint_args)
model = M.Model(args)
if mpi.is_master_node():
    print(model.report())

jperp_tau, d0 = model.spin_couplings(half_prefactor_action=False)
use_jperp, use_szsz = args.jperp != 0, args.szsz != 0


def dlr_imfreq_from_tau(data):
    """Tau-sampled array -> DLR-imfreq Gf, which is how CTINT wants its retarded inputs."""
    g_tau = kernels.as_gf(data, model.beta, target_shape=(1, 1))
    return make_gf_dlr_imfreq(fit_gf_dlr(g_tau, w_max=args.dlr_wmax, eps=args.dlr_eps, symmetrize=True))


S = Solver(beta=model.beta, gf_struct=M.GF_STRUCT, n_tau=model.n_tau_bosonic,
           use_Jperp=use_jperp, use_D=use_szsz, dlr_wmax=args.dlr_wmax)

# G0 from the model's mu and Delta, as common/selfenergy.py builds it
mesh = baths.imfreq_mesh(model.beta, model.n_iw)
g0_inv = selfenergy.g0_inverse_iw(mesh, model.mu, model.delta_iw())
g0_iw = Gf(mesh=mesh, target_shape=(1, 1))
g0_iw << inverse(g0_inv)
g0_dlr = make_gf_dlr_imfreq(fit_gf_dlr(make_gf_from_fourier(g0_iw, model.n_tau),
                                       w_max=args.dlr_wmax, eps=args.dlr_eps, symmetrize=True))
for _, g0_block in S.G0_iw:
    g0_block.data[:, 0, 0] = g0_dlr.data[:, 0, 0]

if use_jperp:
    S.Jperp_iw.data[:] = dlr_imfreq_from_tau(jperp_tau).data[:]
if use_szsz:
    for (s1, s2), d in d0.items():
        S.D0_iw[s1, s2].data[:] = dlr_imfreq_from_tau(d).data[:]


# The D0 channels in the library's (block1, block2) order, for the alpha shifts of common/ctint.py
names = [bl for bl, _ in M.GF_STRUCT]
d0_channels = [d0[bl1, bl2] for bl1 in names for bl2 in names] if use_szsz else []
alpha_kwargs, alpha_report = ctint.alpha_kwargs(args, model.h_int(), d0_channels)
if mpi.is_master_node():
    print(alpha_report)

S.solve(h_int=model.h_int(), **alpha_kwargs,
        length_cycle=args.length_cycle,
        n_warmup_cycles=args.n_warmup_cycles, n_cycles=args.n_cycles, max_time=args.max_time,
        measure_M_iw=True, measure_M_tau=False,
        # On the shared tau grid (library default 201), to compare point by point with CTHYB and CTSEG
        measure_chiAB_tau=True, chi_A_vec=[M.SZ], chi_B_vec=[M.SZ], n_tau_chi2=model.n_tau,
        post_process=True)


def to_uniform_tau(g_iw, n_tau):
    """CTINT returns DLR meshes in some versions and regular ones in others."""
    if isinstance(g_iw.mesh, MeshDLRImFreq):
        return make_gf_imtime(make_gf_dlr(g_iw), n_tau)
    return make_gf_from_fourier(g_iw, n_tau)


def to_regular_imfreq(g_iw, mesh):
    """Bring a possibly-DLR G(iw) onto `mesh`, so the shared Sigma code can use it."""
    if not isinstance(g_iw.mesh, MeshDLRImFreq):
        return g_iw
    out = Gf(mesh=mesh, target_shape=g_iw.target_shape)
    dlr = make_gf_dlr(g_iw)
    for w in mesh:
        out[w] = dlr(w)
    return out


if mpi.is_master_node():
    mu, delta_block = model.sigma_inputs()

    G_tau = {bl: to_uniform_tau(S.G_iw[bl], model.n_tau) for bl, _ in M.GF_STRUCT}
    G_iw_reg = BlockGf(name_list=[bl for bl, _ in M.GF_STRUCT],
                       block_list=[to_regular_imfreq(S.G_iw[bl], mesh) for bl, _ in M.GF_STRUCT])

    sigma_dyson = selfenergy.sigma_from_G_iw(G_iw_reg, mu, delta_block)
    w_n, sigma_up = selfenergy.positive_frequency_part(sigma_dyson["up"])

    # CTINT's own Sigma (Hartree-Fock + M-matrix post-processing) as the second estimator
    sigma_up_alt = None
    if getattr(S, "Sigma_iw", None) is not None:
        sigma_own = to_regular_imfreq(S.Sigma_iw["up"], mesh)
        _, sigma_up_alt = selfenergy.positive_frequency_part(sigma_own)

    density = selfenergy.density_from_G_iw(G_iw_reg)
    sz_mean = 0.5 * (density[M.SPIN_INDEX["up"]] - density[M.SPIN_INDEX["down"]])

    # DLR represents only the connected correlator: remove <Sz>^2 before interpolating, then add it back
    chi_tau = S.chiAB_tau
    disconnected = 0.0
    if isinstance(chi_tau.mesh, MeshDLRImTime):
        disconnected = sz_mean ** 2
        chi_connected = chi_tau.copy()
        chi_connected.data[...] -= disconnected
        chi_tau = make_gf_imtime(make_gf_dlr(chi_connected), model.n_tau_bosonic)
    szsz = chi_tau.data.reshape(chi_tau.data.shape[0], -1)[:, 0].real + disconnected

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
