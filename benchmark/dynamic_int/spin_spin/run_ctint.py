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

Run under `triqs/multiorbital`, which now carries cthyb, ctseg and ctint together.
"""
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
from common import baths, kernels, selfenergy  # noqa: E402

DLR_WMAX = 10.0
DLR_EPS = 1e-10


def add_ctint_args(parser):
    parser.add_argument("--dlr_wmax", type=float, default=DLR_WMAX, help="DLR frequency cutoff")
    parser.add_argument("--dlr_eps", type=float, default=DLR_EPS, help="DLR accuracy")
    parser.add_argument("--alpha", choices=["signed", "library"], default="signed",
                        help="'signed': explicit alpha tensor, D0 shift direction set by the sign of "
                             "each D0 channel (see signed_alpha). 'library': triqs_ctint's automatic "
                             "Hartree-Fock alpha, which is what every run before 2026-09-25 used")
    parser.add_argument("--alpha_delta", type=float, default=0.51,
                        help="Auxiliary-spin shift delta. Larger improves the sign at the cost of a "
                             "higher perturbation order (the library's own default is 0.1)")
    parser.add_argument("--alpha_center", type=float, default=0.5,
                        help="Centre of the signed alpha shifts. Keep 0.5 at every filling: with "
                             "delta = 0.5 + eta the shifts sit just outside [0, 1] (see signed_alpha)")


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


def signed_alpha(h_int, d0, center, delta):
    """alpha tensor, shape (n_terms + n_D0, 2, 2, n_s = 2), in triqs_ctint's layout.

    Each vertex contributes (n_a - alpha_a)(n_b - alpha_b), alpha = center +- delta over the two
    auxiliary spins. For a repulsive coupling the two shifts must go in *opposite* directions,
    for an attractive one in the *same* direction, or the vertex weights come out negative.
    The library's automatic alpha (Solver.find_alpha_from_HF_solver) applies that rule to the
    static h_int terms but gives every D0 channel the same-direction shift regardless of its
    sign. Here D0_ss' = +-spin_kernel/8 has opposite signs in the same- and opposite-spin
    channels, so one of the two was always shifted the wrong way. This applies the static
    rule to each D0 channel too.

    Centred on 1/2 at every filling, *not* on the density. The equal-time factor n - alpha
    only has a definite sign when alpha lies outside [0, 1], so the shifts belong just below
    0 and just above 1: center 1/2, delta = 1/2 + eta. Centring on the density (as the
    library's Hartree-Fock mode does) puts one shift inside [0, 1] away from half filling:
    at beta = 100, n = 0.75, U only, centre 0.75 gave sign 0.41 and the library's 0.90 +- 0.1
    gave 0.00, against 1.00 at half filling.
    """
    n_terms = len(list(h_int))
    names = [bl for bl, _ in M.GF_STRUCT]
    d0_channels = [(bl1, bl2) for bl1 in names for bl2 in names] if use_szsz else []
    alpha = np.zeros((n_terms + len(d0_channels), 2, 2, 2))
    for s, sgn in enumerate((1, -1)):
        for l, (_, coeff) in enumerate(h_int):
            alpha[l, 0, 0, s] = center - np.sign(coeff) * sgn * delta
            alpha[l, 1, 1, s] = center + sgn * delta
        # Same (block1, block2) order as the library's D0 entries (R = 1 orbital per block).
        for d, channel in enumerate(d0_channels):
            sign = np.sign(d0[channel])
            if not np.all(sign == sign[0]):
                raise ValueError(f"D0{channel} changes sign in tau, so no single shift direction fits")
            alpha[n_terms + d, 0, 0, s] = center - sign[0] * sgn * delta
            alpha[n_terms + d, 1, 1, s] = center + sgn * delta
    return alpha


if args.alpha == "signed":
    alpha_kwargs = dict(alpha=signed_alpha(model.h_int(), d0, args.alpha_center, args.alpha_delta), n_s=2)
    if mpi.is_master_node():
        print(f"  signed alpha (center {args.alpha_center:g}, delta {args.alpha_delta:g}):")
        for s in range(2):
            print(f"    s = {s}: " + "  ".join(f"({a[0, 0, s]:+.3f}, {a[1, 1, s]:+.3f})"
                                              for a in alpha_kwargs["alpha"]))
else:
    alpha_kwargs = dict(delta=args.alpha_delta)

S.solve(h_int=model.h_int(), **alpha_kwargs,
        length_cycle=args.length_cycle,
        n_warmup_cycles=args.n_warmup_cycles, n_cycles=args.n_cycles, max_time=args.max_time,
        measure_M_iw=True, measure_M_tau=False,
        # On the shared grid rather than the library's 201 points, so the correlator
        # compares point by point with CTHYB's and CTSEG's (cost: common/grids.py).
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
