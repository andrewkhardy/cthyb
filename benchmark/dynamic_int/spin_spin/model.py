# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# Single-orbital impurity with a retarded spin-spin interaction: CTHYB and CTSEG against a
# CTINT reference. Shared by run_cthyb.py, run_ctseg.py, run_ctint.py and calibrate_mu.py.
#
#   H_loc  = U n_up n_down - mu (n_up + n_down),    Delta(iw) from common/baths.py
#   S_dyn  = 1/2 int int lambda(tau - tau') [szsz Sz Sz + jperp (S+ S- + S- S+) / 2],
#   lambda = -J Q(tau),  Q > 0 from common/kernels.spin_kernel_Q,
#
# each product being O(tau) O'(tau'). --jperp 1 --szsz 1 is the SU(2) symmetric S.S.
# CTHYB's add_dyn_int takes S_dyn as written. CTSEG and CTINT take Jperp(tau) s+ s- and
# D0_ss'(tau) n_s n_s' (lambda is even, so S+S- and S-S+ merge into one s+ s- term):
#   CTSEG  S = 1/2 int int [...]   ->  Jperp = jperp lambda,      D0_ss' = +-szsz lambda / 4
#   CTINT  S =     int int [...]   ->  Jperp = jperp lambda / 2,  D0_ss' = +-szsz lambda / 8
# with + for s = s'.
#
# S.S is even under particle-hole (Sz -> -Sz) and the bath is symmetric, so half filling is
# mu = U/2. Any other filling needs the calibrated --mu from calibrate_mu.py.

import argparse
import os
import sys

from triqs.gfs import BlockGf
from triqs.operators import c, c_dag, n

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import baths, grids, kernels  # noqa: E402

GF_STRUCT = [("down", 1), ("up", 1)]
SPINS = ("up", "down")
SPIN_INDEX = {bl: i for i, (bl, _) in enumerate(GF_STRUCT)}  # position in GF_STRUCT order

SZ = 0.5 * (n("up", 0) - n("down", 0))
SP = c_dag("up", 0) * c("down", 0)
SM = c_dag("down", 0) * c("up", 0)

DEFAULT_OUT_DIR = "/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin"


def add_model_args(parser):
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature")
    parser.add_argument("--U", type=float, default=4.0, help="Hubbard U")
    parser.add_argument("--J", type=float, default=1.0, help="Spin-spin coupling, lambda = -J Q(tau)")
    parser.add_argument("--jperp", type=float, default=1.0, help="Scale of the spin-flip part (0 or 1)")
    parser.add_argument("--szsz", type=float, default=1.0, help="Scale of the Sz.Sz part (0 or 1)")
    parser.add_argument("--filling", type=float, default=0.5,
                        help="Density per spin-orbital, used as a label; 0.5 without --mu sets mu = U/2")
    parser.add_argument("--mu", type=float, default=None, help="Chemical potential; required unless --filling 0.5")
    parser.add_argument("--bath", choices=["dmft", "semicircular", "discrete"], default="dmft",
                        help="'dmft': pole-fitted DMFT bath. 'semicircular': Bethe lattice. 'discrete': one bath site")
    parser.add_argument("--half_bandwidth", type=float, default=2.0, help="For --bath semicircular")
    parser.add_argument("--V_sq", type=float, default=0.49, help="V^2 for --bath discrete")
    parser.add_argument("--n_iw", type=int, default=grids.N_IW, help="Fermionic Matsubara frequencies")
    parser.add_argument("--n_tau", type=int, default=grids.N_TAU, help="Points of every tau mesh, kernel included")
    parser.add_argument("--n_cycles", type=int, default=500000, help="MC cycles per rank")
    parser.add_argument("--n_warmup_cycles", type=int, default=25000, help="Warmup cycles")
    parser.add_argument("--length_cycle", type=int, default=100, help="Moves per cycle")
    parser.add_argument("--max_time", type=int, default=1200, help="Wall-clock cap in seconds, -1 to disable")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="MC seed, also added to the file name; serial chains need distinct seeds")


def parse_args(description, add_solver_args=None):
    parser = argparse.ArgumentParser(description=description)
    add_model_args(parser)
    if add_solver_args is not None:
        add_solver_args(parser)
    return parser.parse_args()


def is_half_filling(filling):
    return abs(filling - 0.5) < 1e-12


class Model:

    def __init__(self, args):
        self.args = args
        self.beta, self.U, self.J = args.beta, args.U, args.J
        self.jperp, self.szsz = args.jperp, args.szsz
        self.n_iw = int(args.n_iw)
        self.n_tau = args.n_tau
        self.n_tau_bosonic = self.n_tau  # one tau grid for G, the kernel and the correlators
        self.spin_kernel = -self.J * kernels.spin_kernel_Q(self.beta, self.n_tau_bosonic)  # lambda(tau)

        if args.mu is not None:
            self.mu = float(args.mu)
        elif is_half_filling(args.filling):
            self.mu = 0.5 * self.U
        else:
            raise ValueError(f"--filling {args.filling} needs an explicit --mu (see calibrate_mu.py)")

    def spin_couplings(self, half_prefactor_action):
        """(Jperp(tau), {(s, s'): D0_ss'(tau)}) for CTSEG (half_prefactor_action=True) or CTINT (False)."""
        f = 1.0 if half_prefactor_action else 0.5
        jperp_tau = (f * self.jperp) * self.spin_kernel
        d0 = {(s1, s2): (f * self.szsz * (0.25 if s1 == s2 else -0.25)) * self.spin_kernel
              for s1 in SPINS for s2 in SPINS}
        return jperp_tau, d0

    def h_int(self):
        return self.U * n("up", 0) * n("down", 0)

    def h_loc0(self):
        return -self.mu * (n("up", 0) + n("down", 0))

    def delta_iw(self):
        """Hybridization as a (1, 1) Gf."""
        mesh = baths.imfreq_mesh(self.beta, self.n_iw)
        return baths.build_delta(mesh, self.args.bath, half_bandwidth=self.args.half_bandwidth,
                                 v_sq=self.args.V_sq, target_shape=(1, 1))

    def sigma_inputs(self):
        """The input mu and Delta (as a BlockGf) for common/selfenergy.py's Dyson equation."""
        delta = self.delta_iw()
        names = [bl for bl, _ in GF_STRUCT]
        return self.mu, BlockGf(name_list=names, block_list=[delta.copy() for _ in names])

    def output_file(self, solver, tag=""):
        os.makedirs(self.args.out_dir, exist_ok=True)
        name = (f"spin_spin_{solver}_b-{self.beta:g}_n-{self.args.filling:g}"
                f"_J-{self.J:g}_jperp-{self.jperp:g}_szsz-{self.szsz:g}"
                + (f"_{tag}" if tag else "")
                + (f"_seed-{self.args.seed}" if self.args.seed is not None else ""))
        return os.path.join(self.args.out_dir, name + ".h5")

    def seed_kwargs(self):
        return {} if self.args.seed is None else {"random_seed": self.args.seed}

    def report(self):
        label = "half filling" if is_half_filling(self.args.filling) else f"n = {self.args.filling}"
        return (f"spin_spin: beta={self.beta:g} U={self.U:g} J={self.J:g} "
                f"jperp={self.jperp:g} szsz={self.szsz:g} bath={self.args.bath} "
                f"mu={self.mu:.6f} ({label})")
