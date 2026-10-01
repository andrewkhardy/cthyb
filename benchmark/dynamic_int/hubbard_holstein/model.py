# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""Single-orbital Hubbard-Holstein model shared by the CTHYB, CTSEG and CTINT runs.

  H = U n_up n_down - mu (n_up + n_down) + bath
    + omega_0 d^dag d + g (n_up + n_down) (d + d^dag)/sqrt(2 omega_0)

Integrating out the phonon gives D_ab(tau) = g^2 Q(tau) (common/kernels.boson_Q) on every
ordered spin pair, diagonal included. CTSEG and CTHYB take S = (1/2) int int sum_ab D0_ab n_a n_b,
so D0_ab = g^2 Q; CTINT's action has no 1/2, so D0_ab = g^2 Q / 2.

The static part, K'(0) = g^2/(2 omega_0^2) per ordered vertex, puts half filling at
mu = U/2 - g^2/omega_0^2; it is computed with static_shift/half_filling_mu, as the solver does.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import baths, grids, kernels  # noqa: E402

from triqs.gfs import BlockGf  # noqa: E402
from triqs.operators import n  # noqa: E402
from triqs_cthyb.dynamical_interactions import half_filling_mu, kprime_0_boson, static_shift  # noqa: E402

BENCHMARK = "hubbard_holstein"
GF_STRUCT = [("down", 1), ("up", 1)]
SPINS = ("up", "down")
# Linear index in GF_STRUCT order, as static_shift expects.
SPIN_INDEX = {"down": 0, "up": 1}

# The phonon couples to the total charge, so that is the correlator measured.
N_TOT = n("up", 0) + n("down", 0)

DEFAULT_OUT_DIR = "/mnt/home/ahardy/ceph/CTHYB_Data/hubbard_holstein"


def add_model_args(parser):
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature")
    parser.add_argument("--U", type=float, default=4.0, help="Hubbard U")
    parser.add_argument("--g", type=float, default=0.7, help="Electron-phonon coupling")
    parser.add_argument("--omega_0", type=float, default=1.0, help="Phonon frequency")
    parser.add_argument("--filling", type=float, default=0.5,
                        help="Target density per spin-orbital; labels the output")
    parser.add_argument("--mu", type=float, default=None,
                        help="Chemical potential; required away from half filling")
    parser.add_argument("--bath", choices=["dmft", "semicircular", "discrete"], default="semicircular",
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


class Model:

    def __init__(self, args):
        self.args = args
        self.beta, self.U = args.beta, args.U
        self.g, self.omega_0 = args.g, args.omega_0
        self.n_iw, self.n_tau = args.n_iw, args.n_tau
        self.n_tau_bosonic = self.n_tau

        self.Q = kernels.boson_Q(self.beta, self.n_tau_bosonic, self.omega_0)
        self.D = self.g ** 2 * self.Q

        self.mu = self._chemical_potential()

    def d0(self, half_prefactor_action):
        """`{(s, s'): D0_ss'}` on every ordered pair: CTSEG/CTHYB convention if True, CTINT's if False."""
        factor = 1.0 if half_prefactor_action else 0.5
        return {(s1, s2): factor * self.D for s1 in SPINS for s2 in SPINS}

    def dyn_vertices_for_static_shift(self):
        """The vertex list in `(a, b, D_tau)` form for `static_shift`, solver convention."""
        return [(SPIN_INDEX[s1], SPIN_INDEX[s2], d)
                for (s1, s2), d in self.d0(half_prefactor_action=True).items()]

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

    def check_half_filling_mu(self, tol=1e-5):
        """Assert half_filling_mu gives U/2 - g^2/omega_0^2. `tol` allows for the Simpson K'(0) of
        static_shift, whose error grows with omega_0 beta on a fixed tau grid."""
        expected = self.U / 2 - self.g ** 2 / self.omega_0 ** 2
        mu = self.half_filling_mu()
        if abs(mu - expected) > tol:
            raise AssertionError(f"half_filling_mu gave {mu}, expected U/2 - g^2/omega_0^2 = {expected}")
        return dict(mu=mu, expected=expected, kprime_0=kprime_0_boson(self.g ** 2, self.omega_0))

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
        name = (f"{BENCHMARK}_{solver}_b-{self.beta:g}_n-{self.args.filling:g}_g-{self.g:g}_w0-{self.omega_0:g}"
                + (f"_{tag}" if tag else ""))
        return os.path.join(self.args.out_dir, name + ".h5")

    def report(self):
        label = "half filling" if abs(self.args.filling - 0.5) < 1e-12 else f"n = {self.args.filling}"
        lam = self.g ** 2 / (self.omega_0 ** 2 * max(self.U, 1e-30))
        return (f"hubbard_holstein: beta={self.beta:g} U={self.U:g} g={self.g:g} "
                f"omega_0={self.omega_0:g} bath={self.args.bath} mu={self.mu:.6f} ({label}); "
                f"polaron shift g^2/omega_0^2={self.g ** 2 / self.omega_0 ** 2:.4f} "
                f"= {lam:.3f} U")


def parse_args(description, add_solver_args=None):
    parser = argparse.ArgumentParser(description=description)
    add_model_args(parser)
    if add_solver_args is not None:
        add_solver_args(parser)
    return parser.parse_args()
