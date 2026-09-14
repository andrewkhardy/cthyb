# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""
Shared setup for the single-orbital spin-spin benchmark: CTHYB and CTSEG against CTINT.
Everything that must be identical across the three solvers (bath, couplings, output
format) lives here, so the factors of 2 are written down exactly once.

Model
-----
  H_loc  = U n_up n_down - mu (n_up + n_down),   mu = U/2
  bath   : G0(iw) = dmft_loop/i_001/S/G0_iw/up of ../ctint.ref.h5 (beta = 10),
           Delta(iw) = iw + mu - G0(iw)^-1 for CTHYB/CTSEG, G0 itself for CTINT
  S_spin = (1/2) int dtau dtau' spin_kernel(tau - tau') S(tau).S(tau'),
           spin_kernel(tau) = -J Q(tau),  Q = dmft_loop/i_000/Q_tau of ../ctint.ref.h5

Since spin_kernel is even, s+(t)s-(t') and s-(t)s+(t') integrate to the same thing, so
  S.S -> s+(tau) s-(tau') + (1/4) sum_{s,s'} (+1 if s == s' else -1) n_s(tau) n_s'(tau')

Factors of 2, read off each solver's source
-------------------------------------------
  CTSEG : S = (1/2) int int [ Jperp s+ s- + sum_{s,s'} D0_{ss'} n_s n_s' ]
          (doc/guide/step_by_step.rst, "Spin-spin interaction": (1/2) Q s.s is set up as
          Jperp_tau = Q, D0 = +-Q/4)
          -> Jperp = spin_kernel,      D0_{ss'} = +-spin_kernel/4
  CTHYB : the same action as CTSEG, so the same inputs. Jperp_tau becomes an S+S- and an
          S-S+ vertex with Jperp/2 each (dynamical_interactions.cpp,
          expand_Jperp_into_vertices); D0_tau becomes one vertex per ordered (s, s') pair;
          moves/insert_dyn.cpp samples every vertex on tau1 > tau2 only, i.e. half of the
          (tau, tau') square -- together exactly (1/2) int int over the full square.
  CTINT : S = int int [ Jperp s+ s- + sum_{s,s'} D0_{ss'} n_s n_s' ]   (no 1/2)
          (vertex_factories.cpp: tau, tau' uniform on the full [0, beta]^2; D0 amplitude
          -D0 per ordered pair, Jperp amplitude Jperp/2 per S+S- / S-S+ type)
          -> Jperp = spin_kernel/2,    D0_{ss'} = +-spin_kernel/8

The switches `jperp` and `szsz` scale the two pieces independently: (1, 1) is the SU(2)
symmetric S.S coupling, (1, 0) is spin-flip only, (0, 1) is Sz.Sz only.
"""
import argparse
import os

import h5
import numpy as np
from triqs.gfs import Gf, GfImTime, inverse, iOmega_n
from triqs.operators import n

HERE = os.path.dirname(os.path.abspath(__file__))
BATH_FILE = os.path.join(HERE, "..", "ctint.ref.h5")
DEFAULT_OUT_DIR = "/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin_vs_ctint"

BETA = 10.0           # fixed by the bath in BATH_FILE (a beta = 10 DMFT solution)
GF_STRUCT = [("down", 1), ("up", 1)]
SPINS = ("up", "down")
N_TAU = 4096          # fermionic tau points (G_tau; CTHYB O_tau)
N_TAU_BOSONIC = 2001  # must equal the number of tau points of Q_tau in BATH_FILE

SZ = 0.5 * (n("up", 0) - n("down", 0))


def h_int(U):
    return U * n("up", 0) * n("down", 0)


def h_loc0(U):
    return -(U / 2) * (n("up", 0) + n("down", 0))


def parse_args(description, add_solver_args=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--J", type=float, default=1.0, help="Spin-spin coupling; spin_kernel = -J Q(tau)")
    parser.add_argument("--U", type=float, default=4.0, help="Hubbard U (mu = U/2)")
    parser.add_argument("--jperp", type=float, default=1.0, help="Scale of the s+s- part (0 or 1)")
    parser.add_argument("--szsz", type=float, default=1.0, help="Scale of the Sz.Sz part (0 or 1)")
    parser.add_argument("--n_cycles", type=int, default=1000000, help="Number of MC cycles")
    parser.add_argument("--n_warmup_cycles", type=int, default=50000, help="Number of warmup cycles")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory")
    if add_solver_args is not None:
        add_solver_args(parser)
    return parser.parse_args()


def load_bath():
    """G0(iw) (1x1, on the file's own mesh) and Q(tau) (1x1 bosonic, N_TAU_BOSONIC points)."""
    with h5.HDFArchive(BATH_FILE, "r") as archive:
        g0 = archive["dmft_loop/i_001/S/G0_iw/up"]
        q_tau = archive["dmft_loop/i_000/Q_tau"]
    if abs(g0.mesh.beta - BETA) > 1e-12:
        raise ValueError(f"Bath beta {g0.mesh.beta} != {BETA}")
    if len(q_tau.mesh) != N_TAU_BOSONIC:
        raise ValueError(f"Q_tau has {len(q_tau.mesh)} points, expected N_TAU_BOSONIC = {N_TAU_BOSONIC}")
    Q_tau = GfImTime(target_shape=[1, 1], statistic="Boson", beta=BETA, n_points=N_TAU_BOSONIC)
    Q_tau.data[:, 0, 0] = q_tau.data[:, 0, 0]
    return g0, Q_tau


def hybridization(g0, U):
    """Delta(iw) = iw + mu - G0(iw)^-1, mu = U/2, on G0's mesh."""
    inverse_g0 = Gf(mesh=g0.mesh, target_shape=[1, 1])
    inverse_g0 << inverse(g0)
    delta = Gf(mesh=g0.mesh, target_shape=[1, 1])
    delta << iOmega_n + U / 2 - inverse_g0
    return delta


def spin_couplings(J, Q_tau, jperp, szsz, half_prefactor_action):
    """Jperp(tau) and {(s, s'): D0_{ss'}(tau)} for S_spin = (1/2) int int (-J Q) S.S.

    half_prefactor_action=True  : CTSEG and CTHYB, S = (1/2) int int [...]
    half_prefactor_action=False : CTINT,           S =       int int [...]
    See the module docstring for the derivation and the source references.
    """
    spin_kernel = -J * Q_tau
    solver_factor = 1.0 if half_prefactor_action else 0.5
    jperp_tau = (solver_factor * jperp) * spin_kernel
    d0 = {(s1, s2): (solver_factor * szsz * (0.25 if s1 == s2 else -0.25)) * spin_kernel for s1 in SPINS for s2 in SPINS}
    return jperp_tau, d0


def tau_points(g):
    return np.array([float(t) for t in g.mesh])


def histogram_array(h):
    """Perturbation-order histogram as a plain array (None stays None)."""
    if h is None:
        return None
    if not isinstance(h, np.ndarray) and hasattr(h, "data"):
        return np.asarray(h.data, dtype=float)
    return np.asarray(h, dtype=float)


def output_file(args, solver, tag=""):
    os.makedirs(args.out_dir, exist_ok=True)
    name = f"{solver}_J-{args.J:g}_U-{args.U:g}_b-{BETA:g}_jperp-{args.jperp:g}_szsz-{args.szsz:g}{tag}.h5"
    return os.path.join(args.out_dir, name)


def save_results(filename, args, solver, tau_G, G_up, tau_SzSz, SzSz, average_sign=None, pert_order_jperp=None, raw=None):
    """Common output format read by plot_spin_spin_vs_ctint.py: G_up(tau), <Sz(tau)Sz(0)>,
    average sign and the Jperp perturbation-order histogram, plus solver-specific raw objects."""
    with h5.HDFArchive(filename, "w") as archive:
        archive["solver"] = solver
        archive["params"] = {key: value for key, value in vars(args).items() if value is not None}
        archive["beta"] = BETA
        archive["tau_G"] = np.asarray(tau_G)
        archive["G_up"] = np.asarray(G_up)
        archive["tau_SzSz"] = np.asarray(tau_SzSz)
        archive["SzSz"] = np.asarray(SzSz)
        if average_sign is not None:
            archive["average_sign"] = float(np.real(average_sign))
        if pert_order_jperp is not None:
            archive["pert_order_jperp"] = histogram_array(pert_order_jperp)
        for key, value in (raw or {}).items():
            if value is not None:
                archive[key] = value
    print(f"Results saved to {filename}")
