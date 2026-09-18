# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""
Shared model for the single-orbital Hubbard-Holstein benchmark: CTHYB and CTSEG against
CTINT. Everything the three solvers must agree on lives here.

Model
-----
  H = U n_up n_down - mu (n_up + n_down)
    + sum_k [eps_k b_k^dag b_k + V_k (c^dag b_k + h.c.)]          the bath
    + omega_0 d^dag d + g (n_up + n_down) (d + d^dag)/sqrt(2 omega_0)

Integrating out the phonon gives a retarded density-density coupling on *every* ordered
pair of spin-orbitals, diagonal included,

  D_ab(tau) = g^2 Q(tau),   Q(tau) = -cosh(omega_0 (tau - beta/2))/(2 omega_0 sinh(omega_0 beta/2))

(doc notes, Eqs. holstein and Qtau). Unlike the spin-spin benchmark's kernel, this one is a
single closed-form pole, so beta = 100 is obtained by evaluating it at beta = 100 -- there is
nothing to transfer, because omega_0 is a declared parameter rather than something inherited
from a stored solution.

Factors of 2
------------
  CTSEG, CTHYB : S = (1/2) int int sum_{ab} D0_ab n_a n_b   -> D0_ab = g^2 Q
  CTINT        : S =       int int sum_{ab} D0_ab n_a n_b   -> D0_ab = g^2 Q / 2
i.e. the same `half_prefactor_action` convention validated in the spin-spin benchmark.
There is no Jperp here: a Holstein phonon couples to the charge, not the spin.

Chemical potential
------------------
The phonon carries a static part. Per ordered vertex with Kp = K'(0) = g^2/(2 omega_0^2),
`static_shift` gives `level_shift[a] = -Kp` from the diagonal (a, a) vertices and
`W_shift[a,b] = -2 Kp = -g^2/omega_0^2` from the two orderings of each off-diagonal pair,
so particle-hole symmetry lands on

    mu = U/2 - g^2/omega_0^2

which `check_half_filling_mu` asserts against the helpers rather than trusting the algebra.
This is the same structure as the Kanamori+phonon ED model
(`W_ab = kanamori_ab - g_a g_b/omega_0^2`, `mu_a = 0.5 sum_b W_ab - g_a^2/(2 omega_0^2)`).

Do *not* use the `-D(i nu = 0)/2` zero-frequency shortcut that the old `holstein.py` used.
It happens to be valid here, since the single-boson kernel is symmetric about beta/2, but
`test/python/dyn_static_shift.py` shows it is off by >0.1 for an asymmetric kernel, so it is
not a habit worth keeping.

Relation to the old script's parameters: `holstein.py` fed
`D(i nu) = 2 L/omega_0 * omega_0^2/((i nu)^2 - omega_0^2) = 2 L omega_0 Q(i nu)`, i.e. its
`L` maps onto `g^2 = 2 L omega_0`.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import baths, io, kernels  # noqa: E402

from triqs.operators import n  # noqa: E402
from triqs_cthyb.dynamical_interactions import half_filling_mu, kprime_0_boson, static_shift  # noqa: E402

BENCHMARK = "hubbard_holstein"
GF_STRUCT = [("down", 1), ("up", 1)]
SPINS = ("up", "down")
SPIN_INDEX = {"down": 0, "up": 1}

# The phonon couples to the total charge, so that is the correlator to measure.
N_TOT = n("up", 0) + n("down", 0)

DEFAULT_OUT_DIR = "/mnt/home/ahardy/ceph/CTHYB_Data/hubbard_holstein"


def add_model_args(parser):
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature")
    parser.add_argument("--U", type=float, default=4.0, help="Hubbard U")
    parser.add_argument("--g", type=float, default=0.7,
                        help="Electron-phonon coupling; D_ab(tau) = g^2 Q(tau) on every ordered pair")
    parser.add_argument("--omega_0", type=float, default=1.0, help="Phonon frequency")
    parser.add_argument("--filling", type=float, default=0.5,
                        help="Target density per spin-orbital (0.5 = half filling)")
    parser.add_argument("--mu", type=float, default=None,
                        help="Chemical potential; at half filling defaults to the exact U/2 - g^2/omega_0^2")
    parser.add_argument("--bath", choices=["dmft", "semicircular", "discrete"], default="semicircular",
                        help="'semicircular' (default): first-iteration Bethe, the natural clean choice "
                             "for a Holstein benchmark. 'discrete': one bath site, so the model is ED-"
                             "representable. 'dmft': the pole-fitted correlated bath of the spin-spin "
                             "benchmark, for comparing the two")
    parser.add_argument("--half_bandwidth", type=float, default=2.0, help="For --bath semicircular")
    parser.add_argument("--V_sq", type=float, default=0.49, help="V^2 for --bath discrete")
    parser.add_argument("--n_iw", type=int, default=1025, help="Fermionic Matsubara frequencies")
    parser.add_argument("--n_tau", type=int, default=4096, help="Fermionic tau points")
    parser.add_argument("--n_tau_bosonic", type=int, default=2001, help="Bosonic tau points")
    parser.add_argument("--n_cycles", type=int, default=500000, help="MC cycles (halved from the old 1e6)")
    parser.add_argument("--n_warmup_cycles", type=int, default=25000, help="Warmup cycles")
    parser.add_argument("--length_cycle", type=int, default=100, help="Moves per cycle")
    parser.add_argument("--max_time", type=int, default=1200,
                        help="Hard wall-clock cap in seconds for the MC, -1 to disable")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory")


class Model:

    def __init__(self, args):
        self.args = args
        self.beta, self.U = args.beta, args.U
        self.g, self.omega_0 = args.g, args.omega_0
        self.n_iw, self.n_tau, self.n_tau_bosonic = args.n_iw, args.n_tau, args.n_tau_bosonic

        # Single Einstein boson, closed form, valid at any beta.
        self.Q = kernels.boson_Q(self.beta, self.n_tau_bosonic, self.omega_0)
        self.D = self.g ** 2 * self.Q

        self.mu = self._chemical_potential()

    # ------------------------------------------------------------------ couplings

    def d0(self, half_prefactor_action):
        """`{(s, s'): D0_ss'}` for every ordered pair, diagonal included."""
        factor = 1.0 if half_prefactor_action else 0.5
        return {(s1, s2): factor * self.D for s1 in SPINS for s2 in SPINS}

    def dyn_vertices_for_static_shift(self):
        """Vertex list in `(a, b, D_tau)` form for `static_shift`, solver convention."""
        return [(SPIN_INDEX[s1], SPIN_INDEX[s2], d)
                for (s1, s2), d in self.d0(half_prefactor_action=True).items()]

    # ------------------------------------------------------------------ mu

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
        mu = half_filling_mu(W_static, W_shift, level_shift)
        if not np.allclose(mu, mu[0]):
            raise AssertionError(f"mu should be spin-independent, got {mu}")
        return mu[0]

    def check_half_filling_mu(self, tol=1e-5):
        """Assert the helpers reproduce `U/2 - g^2/omega_0^2`, and return the pieces.

        `tol` is loose on purpose. `static_shift` gets K'(0) from `kprime_0`, which is
        Simpson quadrature on the `n_tau_bosonic` grid, and the boson kernel is most
        strongly curved exactly at tau = 0 and beta -- so the error grows with the
        peakedness `omega_0 beta / 2` and falls as n_tau^-4. Measured relative error at
        n_tau_bosonic = 2001: 3e-12 at omega_0 beta/2 = 5, 3e-8 at 50, 6e-7 at 100,
        9e-6 at 200 (8001 points buys ~250x at each).

        Using the exact `kprime_0_boson` closed form here would make the check trivially
        tight, but it would also stop testing the code path that matters: the solver
        computes its own shift numerically, so mu should be built the same way it is.
        Consequently a large `omega_0 * beta` wants a finer bosonic grid -- see the note in
        the submit script.
        """
        expected = self.U / 2 - self.g ** 2 / self.omega_0 ** 2
        W_shift, level_shift = static_shift(self.dyn_vertices_for_static_shift(), 2, self.beta)
        mu = self.half_filling_mu()
        if abs(mu - expected) > tol:
            raise AssertionError(
                f"Expected mu = U/2 - g^2/omega_0^2 = {expected} for the Holstein model, but "
                f"half_filling_mu gave {mu}. W_shift =\n{W_shift}\nlevel_shift = {level_shift}")
        return dict(mu=mu, expected=expected, W_shift=W_shift, level_shift=level_shift,
                    kprime_0=kprime_0_boson(self.g ** 2, self.omega_0))

    # ------------------------------------------------------------------ operators / bath

    def h_int(self):
        return self.U * n("up", 0) * n("down", 0)

    def h_loc0(self):
        return -self.mu * (n("up", 0) + n("down", 0))

    def delta_iw(self):
        mesh = baths.imfreq_mesh(self.beta, self.n_iw)
        return baths.build_delta(mesh, self.args.bath, half_bandwidth=self.args.half_bandwidth,
                                 v_sq=self.args.V_sq, target_shape=(1, 1))

    def delta_iw_block(self):
        from triqs.gfs import BlockGf
        delta = self.delta_iw()
        names = [name for name, _ in GF_STRUCT]
        return BlockGf(name_list=names, block_list=[delta.copy() for _ in names])

    def sigma_inputs(self):
        """The *input* mu and Delta; never read back from a solver whose h_loc was shifted."""
        return self.mu, self.delta_iw_block()

    # ------------------------------------------------------------------ bookkeeping

    def params(self):
        return dict(beta=self.beta, U=self.U, g=self.g, omega_0=self.omega_0, mu=self.mu,
                    filling=self.args.filling, bath=self.args.bath, n_iw=self.n_iw,
                    n_tau=self.n_tau, n_tau_bosonic=self.n_tau_bosonic)

    def output_file(self, solver, tag=""):
        return io.output_file(self.args.out_dir, BENCHMARK, solver, self.beta, self.args.filling,
                              tag=f"g-{self.g:g}_w0-{self.omega_0:g}" + (f"_{tag}" if tag else ""))

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
