# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""Single-orbital Hubbard-Holstein impurity, shared by run_cthyb.py, run_ctseg.py,
run_ctint.py and calibrate_mu.py:

    H = U n_up n_down - mu N + H_bath + omega_0 d^dag d + g N (d + d^dag) / sqrt(2 omega_0),
    N = n_up + n_down, H_bath set by --bath.

Integrating out the phonon gives the retarded interaction

    S_dyn = 1/2 int int g^2 Q(tau - tau') N(tau) N(tau'),
    Q(tau) = -cosh(omega_0 (tau - beta/2)) / (2 omega_0 sinh(omega_0 beta/2)).
"""
import argparse
import os
import sys

from triqs.gfs import BlockGf
from triqs.operators import n
from triqs_cthyb.dynamical_interactions import kprime_0

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import baths, grids, kernels  # noqa: E402

GF_STRUCT = [("down", 1), ("up", 1)]
SPINS = ("up", "down")
N_TOT = n("up", 0) + n("down", 0)  # what the phonon couples to


def add_model_args(parser):
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature")
    parser.add_argument("--U", type=float, default=4.0, help="Hubbard U")
    parser.add_argument("--g", type=float, default=0.7, help="Electron-phonon coupling")
    parser.add_argument("--omega_0", type=float, default=1.0, help="Phonon frequency")
    parser.add_argument("--filling", type=float, default=0.5,
                        help="Density per spin-orbital, used in the file name; away from 0.5, pass --mu")
    parser.add_argument("--mu", type=float, default=None, help="Chemical potential; default half filling")
    parser.add_argument("--bath", choices=["dmft", "semicircular", "discrete"], default="semicircular",
                        help="'semicircular': first-iteration Bethe; 'discrete': one bath site; "
                             "'dmft': the pole-fitted bath of the spin-spin benchmark")
    parser.add_argument("--half_bandwidth", type=float, default=2.0, help="For --bath semicircular")
    parser.add_argument("--V_sq", type=float, default=0.49, help="V^2 for --bath discrete")
    parser.add_argument("--n_iw", type=int, default=grids.N_IW, help="Fermionic Matsubara frequencies")
    parser.add_argument("--n_tau", type=int, default=grids.N_TAU, help="Tau points for G, the kernel and correlators")
    # Monte Carlo, shared by every solver
    parser.add_argument("--n_cycles", type=int, default=500000)
    parser.add_argument("--n_warmup_cycles", type=int, default=25000)
    parser.add_argument("--length_cycle", type=int, default=100)
    parser.add_argument("--max_time", type=int, default=1200, help="Wall-clock cap in seconds, -1 to disable")
    parser.add_argument("--out_dir", default="/mnt/home/ahardy/ceph/CTHYB_Data/hubbard_holstein")


def parse_args(description, add_solver_args=None):
    parser = argparse.ArgumentParser(description=description)
    add_model_args(parser)
    if add_solver_args is not None:
        add_solver_args(parser)
    return parser.parse_args()


class Model:

    def __init__(self, args):
        self.args = args
        self.beta, self.U, self.g, self.omega_0 = args.beta, args.U, args.g, args.omega_0
        self.n_iw, self.n_tau = args.n_iw, args.n_tau
        self.n_tau_bosonic = self.n_tau
        self.Q = kernels.boson_Q(self.beta, self.n_tau_bosonic, self.omega_0)
        self.mu = self._chemical_potential()

    def _chemical_potential(self):
        if self.args.mu is not None:
            return float(self.args.mu)
        if abs(self.args.filling - 0.5) > 1e-12:
            raise ValueError(f"--filling {self.args.filling} has no closed-form mu: pass --mu (see calibrate_mu.py)")
        # The phonon's static part -K'(0) N^2, with K'(0) = g^2 / (2 omega_0^2), turns U into
        # U - 2 K'(0) and shifts the level by -K'(0). Half filling is then mu = U/2 - g^2/omega_0^2.
        # kprime_0 integrates the sampled kernel, so this matches the closed form to quadrature accuracy.
        kp = kprime_0(self.g ** 2 * self.Q, self.beta)
        return float(0.5 * (self.U - 2 * kp) - kp)

    def d0(self, half_prefactor_action):
        """D0_ss' on every ordered spin pair, diagonal included, for CTSEG's and CTINT's D0 inputs.

        CTSEG's action carries the 1/2 of S_dyn, so D0 = g^2 Q; CTINT's does not, so D0 = g^2 Q / 2.
        """
        factor = 1.0 if half_prefactor_action else 0.5
        return {(s1, s2): factor * self.g ** 2 * self.Q for s1 in SPINS for s2 in SPINS}

    def h_int(self):
        return self.U * n("up", 0) * n("down", 0)

    def h_loc0(self):
        return -self.mu * N_TOT

    def delta_iw(self):
        """Hybridization of one spin, a (1, 1) Gf."""
        mesh = baths.imfreq_mesh(self.beta, self.n_iw)
        return baths.build_delta(mesh, self.args.bath, half_bandwidth=self.args.half_bandwidth,
                                 v_sq=self.args.V_sq, target_shape=(1, 1))

    def sigma_inputs(self):
        """The input mu and Delta (as a BlockGf) for Dyson's equation."""
        delta = self.delta_iw()
        names = [name for name, _ in GF_STRUCT]
        return self.mu, BlockGf(name_list=names, block_list=[delta.copy() for _ in names])

    def output_file(self, solver, tag=""):
        os.makedirs(self.args.out_dir, exist_ok=True)
        name = (f"hubbard_holstein_{solver}_b-{self.beta:g}_n-{self.args.filling:g}_g-{self.g:g}_w0-{self.omega_0:g}"
                + (f"_{tag}" if tag else ""))
        return os.path.join(self.args.out_dir, name + ".h5")

    def report(self):
        label = "half filling" if abs(self.args.filling - 0.5) < 1e-12 else f"n = {self.args.filling}"
        shift = self.g ** 2 / self.omega_0 ** 2
        return (f"hubbard_holstein: beta={self.beta:g} U={self.U:g} g={self.g:g} omega_0={self.omega_0:g} "
                f"bath={self.args.bath} mu={self.mu:.6f} ({label}); "
                f"polaron shift g^2/omega_0^2={shift:.4f} = {shift / max(self.U, 1e-30):.3f} U")
