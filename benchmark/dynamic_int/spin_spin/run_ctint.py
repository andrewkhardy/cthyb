# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""CTINT reference for the single-orbital spin-spin benchmark (model and conventions: model.py).

CTINT's action has no 1/2 in front of the retarded terms, so its Jperp and D0 are half of
CTSEG's and CTHYB's -- `spin_couplings(..., half_prefactor_action=False)`.

CTINT takes `G0_iw` rather than `Delta_tau`, and it is handed exactly the
`G0^-1 = iw + mu - Delta` that `common/selfenergy.py` uses, so the chemical-potential
convention is shared by construction rather than by coincidence.

Run with the CTINT module stack (`modules/2.4 ... triqs/3_unst_nix2.4_llvm`), which is
incompatible with the CTHYB and CTSEG ones.
"""
import os
import sys

import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import (BlockGf, Gf, MeshDLRImFreq, MeshDLRImTime, fit_gf_dlr, inverse, make_gf_dlr,
                       make_gf_dlr_imfreq, make_gf_from_fourier, make_gf_imtime)
from triqs_ctint import Solver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402
from common import baths, io, kernels, selfenergy  # noqa: E402

DLR_WMAX = 10.0
DLR_EPS = 1e-10


def add_ctint_args(parser):
    parser.add_argument("--dlr_wmax", type=float, default=DLR_WMAX, help="DLR frequency cutoff")
    parser.add_argument("--dlr_eps", type=float, default=DLR_EPS, help="DLR accuracy")


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

# G0 from the model's own mu and Delta -- the same object common/selfenergy.py builds.
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

S.solve(h_int=model.h_int(),
        n_warmup_cycles=args.n_warmup_cycles, n_cycles=args.n_cycles, max_time=args.max_time,
        measure_M_iw=True, measure_M_tau=False,
        measure_chiAB_tau=True, chi_A_vec=[M.SZ], chi_B_vec=[M.SZ],
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

    # CTINT's own Sigma, via its Hartree-Fock + M-matrix post-processing, as the second route.
    sigma_up_alt = None
    if getattr(S, "Sigma_iw", None) is not None:
        sigma_own = to_regular_imfreq(S.Sigma_iw["up"], mesh)
        _, sigma_up_alt = selfenergy.positive_frequency_part(sigma_own)

    density = selfenergy.density_from_G_iw(G_iw_reg)
    sz_mean = 0.5 * (density[M.SPIN_INDEX["up"]] - density[M.SPIN_INDEX["down"]])

    # The DLR representation holds for the *connected* correlator only, so take out <Sz>^2
    # before interpolating and add it back afterwards.
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
    print(f"  average sign = {S.average_sign:.4f}   <n> = {np.round(density, 5)}")

    io.save(model.output_file("ctint"),
            solver="ctint", params=vars(args), beta=model.beta, mu=mu,
            tau_G=np.array([float(t) for t in G_tau["up"].mesh]), G=G_tau["up"].data[:, 0, 0].real,
            w_n=w_n, Sigma=sigma_up, Sigma_alt=sigma_up_alt,
            tau_corr=np.array([float(t) for t in chi_tau.mesh]), corr=szsz,
            corr_label=r"$\langle S_z(\tau)S_z(0)\rangle$ (chiAB)",
            density=density, average_sign=S.average_sign,
            raw={"G_iw": S.G_iw, "chiAB_tau": S.chiAB_tau, "G0_iw": S.G0_iw,
                 "Sigma_iw_dyson": sigma_dyson})
