# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""Single-orbital retarded spin-spin model shared by the CTHYB, CTSEG and CTINT runs.

  H_loc  = U n_up n_down - mu (n_up + n_down),   bath from common/baths.py
  S_spin = (1/2) int dtau dtau' spin_kernel(tau - tau') S(tau).S(tau'),
           spin_kernel(tau) = -J Q(tau),  Q = common/kernels.spin_kernel_Q

spin_kernel is even, so S.S -> s+(tau) s-(tau') + (1/4) sum_{s,s'} (+1 if s == s' else -1) n_s n_s'.

Factors of 2, read off each solver's source:
  CTSEG : S = (1/2) int int [ Jperp s+ s- + sum_{s,s'} D0_{ss'} n_s n_s' ]
          (doc/guide/step_by_step.rst)        -> Jperp = spin_kernel,   D0_{ss'} = +-spin_kernel/4
  CTHYB : the same inputs. Jperp_tau becomes S+S- and S-S+ vertices with Jperp/2 each
          (expand_Jperp_into_vertices), D0_tau one vertex per ordered (s, s'), and
          moves/insert_dyn.cpp samples tau1 > tau2 only: together (1/2) int int.
  CTINT : S = int int [ Jperp s+ s- + sum_{s,s'} D0_{ss'} n_s n_s' ]   (no 1/2, vertex_factories.cpp)
                                              -> Jperp = spin_kernel/2, D0_{ss'} = +-spin_kernel/8

`--jperp` and `--szsz` scale the two pieces: (1, 1) is SU(2) symmetric S.S.

The Sz.Sz part carries a static K'(0), but S.S is even under particle-hole and the bath is
symmetric, so half filling is exactly mu = U/2; it is computed with half_filling_mu
(level_shift enters with a plus) and checked against U/2. Away from half filling --mu is
required.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import baths, grids, kernels  # noqa: E402

from triqs.gfs import BlockGf  # noqa: E402
from triqs.operators import n  # noqa: E402
from triqs_cthyb.dynamical_interactions import half_filling_mu, static_shift  # noqa: E402

BENCHMARK = "spin_spin"
GF_STRUCT = [("down", 1), ("up", 1)]
SPINS = ("up", "down")
# Linear index in GF_STRUCT order, as static_shift expects.
SPIN_INDEX = {"down": 0, "up": 1}

SZ = 0.5 * (n("up", 0) - n("down", 0))

DEFAULT_OUT_DIR = "/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin"


def add_model_args(parser):
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature")
    parser.add_argument("--U", type=float, default=4.0, help="Hubbard U")
    parser.add_argument("--J", type=float, default=1.0, help="Spin-spin coupling; spin_kernel = -J Q(tau)")
    parser.add_argument("--jperp", type=float, default=1.0, help="Scale of the s+s- part (0 or 1)")
    parser.add_argument("--szsz", type=float, default=1.0, help="Scale of the Sz.Sz part (0 or 1)")
    parser.add_argument("--filling", type=float, default=0.5,
                        help="Target density per spin-orbital; labels the output")
    parser.add_argument("--mu", type=float, default=None,
                        help="Chemical potential; required away from half filling")
    parser.add_argument("--bath", choices=["dmft", "semicircular", "discrete"], default="dmft",
                        help="Hybridization, see common/baths.py")
    parser.add_argument("--half_bandwidth", type=float, default=2.0, help="For --bath semicircular")
    parser.add_argument("--V_sq", type=float, default=0.49, help="V^2 for --bath discrete")
    parser.add_argument("--n_iw", type=int, default=grids.N_IW, help="Fermionic Matsubara frequencies")
    parser.add_argument("--n_tau", type=int, default=grids.N_TAU, help="Tau points of every tau quantity")
    parser.add_argument("--n_cycles", type=int, default=500000, help="MC cycles per rank")
    parser.add_argument("--n_warmup_cycles", type=int, default=25000, help="Warmup cycles")
    parser.add_argument("--length_cycle", type=int, default=100, help="Moves per cycle")
    parser.add_argument("--max_time", type=int, default=1200, help="MC wall-clock cap in seconds, -1 for none")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="MC random seed, also appended to the file name (default: the solver's)")


class Model:

    def __init__(self, args):
        self.args = args
        self.beta, self.U, self.J = args.beta, args.U, args.J
        self.jperp, self.szsz = args.jperp, args.szsz
        self.n_iw = int(args.n_iw)
        self.n_tau = args.n_tau
        self.n_tau_bosonic = self.n_tau

        self.Q = kernels.spin_kernel_Q(self.beta, self.n_tau_bosonic)
        self.spin_kernel = -self.J * self.Q

        self.mu = self._chemical_potential()

    def spin_couplings(self, half_prefactor_action):
        """`(jperp_tau, {(s, s'): D0_ss'})`: CTSEG/CTHYB convention if True, CTINT's if False."""
        solver_factor = 1.0 if half_prefactor_action else 0.5
        jperp_tau = (solver_factor * self.jperp) * self.spin_kernel
        d0 = {(s1, s2): (solver_factor * self.szsz * (0.25 if s1 == s2 else -0.25)) * self.spin_kernel
              for s1 in SPINS for s2 in SPINS}
        return jperp_tau, d0

    def dyn_vertices_for_static_shift(self):
        """The Sz.Sz vertex list in `(a, b, D_tau)` form for `static_shift`, solver convention."""
        _, d0 = self.spin_couplings(half_prefactor_action=True)
        return [(SPIN_INDEX[s1], SPIN_INDEX[s2], d) for (s1, s2), d in d0.items()]

    def _chemical_potential(self):
        if self.args.mu is not None:
            return float(self.args.mu)
        if abs(self.args.filling - 0.5) > 1e-12:
            raise ValueError(
                f"--filling {self.args.filling} is away from half filling, which has no closed-form "
                "mu: pass --mu explicitly (the submit script carries the calibrated values)")
        return float(self.half_filling_mu())

    def half_filling_mu(self):
        W_static = np.array([[0.0, self.U], [self.U, 0.0]])
        W_shift, level_shift = static_shift(self.dyn_vertices_for_static_shift(), 2, self.beta)
        return half_filling_mu(W_static, W_shift, level_shift)[0]

    def check_half_filling_mu(self, tol=1e-10):
        """Assert half_filling_mu gives U/2, and return the pieces a run reports."""
        W_shift, level_shift = static_shift(self.dyn_vertices_for_static_shift(), 2, self.beta)
        mu = self.half_filling_mu()
        if abs(mu - self.U / 2) > tol:
            raise AssertionError(
                f"Particle-hole symmetry requires mu = U/2 = {self.U / 2} for this model, but "
                f"half_filling_mu gave {mu}. W_shift =\n{W_shift}\nlevel_shift = {level_shift}")
        return dict(mu=mu, W_shift=W_shift, level_shift=level_shift)

    def h_int(self):
        return self.U * n("up", 0) * n("down", 0)

    def h_loc0(self):
        return -self.mu * (n("up", 0) + n("down", 0))

    def delta_iw(self):
        mesh = baths.imfreq_mesh(self.beta, self.n_iw)
        return baths.build_delta(mesh, self.args.bath, half_bandwidth=self.args.half_bandwidth,
                                 v_sq=self.args.V_sq, target_shape=(1, 1))

    def delta_iw_block(self):
        delta = self.delta_iw()
        names = [name for name, _ in GF_STRUCT]
        return BlockGf(name_list=names, block_list=[delta.copy() for _ in names])

    def sigma_inputs(self):
        """The *input* mu and Delta; never read back from a solver whose h_loc was shifted."""
        return self.mu, self.delta_iw_block()

    def output_file(self, solver, tag=""):
        os.makedirs(self.args.out_dir, exist_ok=True)
        name = (f"{BENCHMARK}_{solver}_b-{self.beta:g}_n-{self.args.filling:g}"
                f"_J-{self.J:g}_jperp-{self.jperp:g}_szsz-{self.szsz:g}"
                + (f"_{tag}" if tag else "")
                + (f"_seed-{self.args.seed}" if self.args.seed is not None else ""))
        return os.path.join(self.args.out_dir, name + ".h5")

    def seed_kwargs(self):
        """`random_seed` for solve(), only when --seed was given."""
        return {} if self.args.seed is None else {"random_seed": self.args.seed}

    def report(self):
        label = "half filling" if abs(self.args.filling - 0.5) < 1e-12 else f"n = {self.args.filling}"
        return (f"spin_spin: beta={self.beta:g} U={self.U:g} J={self.J:g} "
                f"jperp={self.jperp:g} szsz={self.szsz:g} bath={self.args.bath} "
                f"mu={self.mu:.6f} ({label})")


def parse_args(description, add_solver_args=None):
    parser = argparse.ArgumentParser(description=description)
    add_model_args(parser)
    if add_solver_args is not None:
        add_solver_args(parser)
    return parser.parse_args()
