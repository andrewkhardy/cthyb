# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""CTINT reference for the single-orbital spin-spin benchmark (model and conventions: spin_spin_common.py).

CTINT's action has no 1/2 in front of the retarded terms, so its Jperp and D0 are half of
CTSEG's/CTHYB's (spin_couplings(..., half_prefactor_action=False))."""
import triqs.utility.mpi as mpi
from triqs.gfs import fit_gf_dlr, make_gf_dlr, make_gf_dlr_imfreq, make_gf_from_fourier, make_gf_imtime
from triqs_ctint import Solver

import spin_spin_common as common

DLR_WMAX = 10.0
DLR_EPS = 1e-10

args = common.parse_args("CTINT single-orbital spin-spin benchmark (reference)")
g0, Q_tau = common.load_bath()
jperp_tau, d0 = common.spin_couplings(args.J, Q_tau, args.jperp, args.szsz, half_prefactor_action=False)
use_jperp = args.jperp != 0
use_szsz = args.szsz != 0


def dlr_imfreq(g_tau):
    return make_gf_dlr_imfreq(fit_gf_dlr(g_tau, w_max=DLR_WMAX, eps=DLR_EPS, symmetrize=True))


S = Solver(beta=common.BETA, gf_struct=common.GF_STRUCT, n_tau=common.N_TAU_BOSONIC,
           use_Jperp=use_jperp, use_D=use_szsz, dlr_wmax=DLR_WMAX)

g0_iw = dlr_imfreq(make_gf_from_fourier(g0))
for _, g0_block in S.G0_iw:
    g0_block.data[:, 0, 0] = g0_iw.data[:, 0, 0]
if use_jperp:
    S.Jperp_iw.data[:] = dlr_imfreq(jperp_tau).data[:]
if use_szsz:
    for (s1, s2), d in d0.items():
        S.D0_iw[s1, s2].data[:] = dlr_imfreq(d).data[:]

S.solve(h_int=common.h_int(args.U), n_warmup_cycles=args.n_warmup_cycles, n_cycles=args.n_cycles,
        measure_M_iw=True, measure_M_tau=False, measure_chiAB_tau=True, chi_ops=[(common.SZ, common.SZ)],
        post_process=True)

if mpi.is_master_node():
    # G_iw and chiAB_tau live on DLR meshes: evaluate on uniform tau grids.
    G_tau = {bl: make_gf_imtime(make_gf_dlr(S.G_iw[bl]), common.N_TAU) for bl in ("up", "down")}
    densities = {bl: -G_tau[bl].data[-1, 0, 0].real for bl in G_tau}
    # The DLR representation holds for the connected correlator only: take out <Sz>^2 before
    # interpolating, add it back after (as in compare_ctint_ctseg.py).
    Sz_mean = 0.5 * (densities["up"] - densities["down"])
    chi_connected = S.chiAB_tau.copy()
    chi_connected.data[:, 0] -= Sz_mean**2
    chi_tau = make_gf_imtime(make_gf_dlr(chi_connected), common.N_TAU_BOSONIC)
    common.save_results(common.output_file(args, "ctint"), args, "ctint",
                        common.tau_points(G_tau["up"]), G_tau["up"].data[:, 0, 0].real,
                        common.tau_points(chi_tau), chi_tau.data[:, 0].real + Sz_mean**2,
                        average_sign=S.average_sign,
                        raw={"G_iw": S.G_iw, "chiAB_tau": S.chiAB_tau, "G0_iw": S.G0_iw})
