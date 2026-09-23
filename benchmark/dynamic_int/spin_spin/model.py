# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""
Shared model for the single-orbital retarded spin-spin benchmark: CTHYB and CTSEG against
CTINT. Everything that must be identical across the three solvers -- bath, couplings,
chemical potential, output format -- lives here, so the factors of 2 are written down once.

Model
-----
  H_loc  = U n_up n_down - mu (n_up + n_down)
  bath   : Delta(iw) = sum_k V_k^2/(iw - eps_k), particle-hole symmetric (common/baths.py)
  S_spin = (1/2) int dtau dtau' spin_kernel(tau - tau') S(tau).S(tau'),
           spin_kernel(tau) = -J Q(tau),  Q from common/kernels.spin_kernel_Q

Since `spin_kernel` is even, s+(t)s-(t') and s-(t)s+(t') integrate to the same thing, so

  S.S -> s+(tau) s-(tau') + (1/4) sum_{s,s'} (+1 if s == s' else -1) n_s(tau) n_s'(tau')

Factors of 2, read off each solver's source
-------------------------------------------
  CTSEG : S = (1/2) int int [ Jperp s+ s- + sum_{s,s'} D0_{ss'} n_s n_s' ]
          (doc/guide/step_by_step.rst, "Spin-spin interaction")
          -> Jperp = spin_kernel,      D0_{ss'} = +-spin_kernel/4
  CTHYB : the same action as CTSEG, so the same inputs. Jperp_tau becomes an S+S- and an
          S-S+ vertex with Jperp/2 each (dynamical_interactions.cpp,
          expand_Jperp_into_vertices); D0_tau becomes one vertex per ordered (s, s') pair;
          moves/insert_dyn.cpp samples every vertex on tau1 > tau2 only, i.e. half of the
          (tau, tau') square -- together exactly (1/2) int int over the full square.
  CTINT : S = int int [ Jperp s+ s- + sum_{s,s'} D0_{ss'} n_s n_s' ]   (no 1/2)
          (vertex_factories.cpp)
          -> Jperp = spin_kernel/2,    D0_{ss'} = +-spin_kernel/8

`--jperp` and `--szsz` scale the two pieces independently: (1, 1) is the SU(2) symmetric
S.S coupling, (1, 0) spin-flip only, (0, 1) Sz.Sz only.

Chemical potential
------------------
The retarded Sz.Sz coupling carries a static K'(0) part, so mu is *not* simply U/2 term by
term -- but the full S.S interaction is particle-hole symmetric (S_z -> -S_z under
c <-> c^dag, and S.S is even in S), and the bath is symmetric by construction, so the two
contributions cancel and mu = U/2 exactly. `half_filling_mu` is used rather than the
literal U/2, and `check_half_filling_mu` asserts they agree -- a free consistency check on
the sign conventions, since `level_shift` enters mu with a *plus* and getting that backwards
is easy. (Verified: W_shift = +0.381 off-diagonal against level_shift = -0.1905 at
U = 4, J = 1, beta = 10, cancelling to give exactly 2.0.)

Away from half filling there is no closed form, so `--mu` must be given explicitly; the
submit script carries the calibrated values so every solver uses the identical mu, which is
what makes the Sigma comparison meaningful.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import baths, io, kernels, selfenergy  # noqa: E402

from triqs.operators import n  # noqa: E402
from triqs_cthyb.dynamical_interactions import half_filling_mu, static_shift  # noqa: E402

BENCHMARK = "spin_spin"
GF_STRUCT = [("down", 1), ("up", 1)]
SPINS = ("up", "down")
# Linear index of each spin-orbital, in itertools.product(block_names, range(size)) order
# -- the order static_shift expects.
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
                        help="Target density per spin-orbital (0.5 = half filling). Only labels the "
                             "output unless it is 0.5, where mu follows from particle-hole symmetry")
    parser.add_argument("--mu", type=float, default=None,
                        help="Chemical potential. Required away from half filling; at half filling "
                             "defaults to the exact particle-hole symmetric value")
    parser.add_argument("--bath", choices=["dmft", "semicircular", "discrete"], default="dmft",
                        help="'dmft': the pole-fitted bath of the original benchmark (default, so "
                             "beta = 10 stays comparable with earlier results). 'semicircular': "
                             "first-iteration Bethe. 'discrete': one bath site, ED-representable")
    parser.add_argument("--half_bandwidth", type=float, default=2.0, help="For --bath semicircular")
    parser.add_argument("--V_sq", type=float, default=0.49, help="V^2 for --bath discrete")
    parser.add_argument("--n_iw", type=float, default=1025, help="Fermionic Matsubara frequencies")
    parser.add_argument("--n_tau", type=int, default=4096, help="Fermionic tau points")
    parser.add_argument("--n_tau_bosonic", type=int, default=2001, help="Bosonic tau points")
    parser.add_argument("--n_cycles", type=int, default=500000, help="MC cycles (halved from the old 1e6)")
    parser.add_argument("--n_warmup_cycles", type=int, default=25000, help="Warmup cycles")
    parser.add_argument("--length_cycle", type=int, default=100, help="Moves per cycle")
    parser.add_argument("--max_time", type=int, default=1200,
                        help="Hard wall-clock cap in seconds for the MC, -1 to disable")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="MC random seed, also appended to the filename. For independent serial "
                             "chains (run_chains.sh): the solvers' default seed depends only on the MPI "
                             "rank, so every serial run would otherwise be the same chain")


class Model:
    """The benchmark model, built once and handed to whichever solver is running."""

    def __init__(self, args):
        self.args = args
        self.beta, self.U, self.J = args.beta, args.U, args.J
        self.jperp, self.szsz = args.jperp, args.szsz
        self.n_iw = int(args.n_iw)
        self.n_tau, self.n_tau_bosonic = args.n_tau, args.n_tau_bosonic

        # Retarded kernel, at this beta. Positive, symmetric about beta/2.
        self.Q = kernels.spin_kernel_Q(self.beta, self.n_tau_bosonic)
        self.spin_kernel = -self.J * self.Q

        self.mu = self._chemical_potential()

    # ------------------------------------------------------------------ couplings

    def spin_couplings(self, half_prefactor_action):
        """`(jperp_tau, {(s, s'): D0_ss'})` as plain arrays, in the given solver's convention.

        `half_prefactor_action=True` for CTSEG and CTHYB (their action carries an explicit
        1/2), False for CTINT. See the module docstring for the derivation.
        """
        solver_factor = 1.0 if half_prefactor_action else 0.5
        jperp_tau = (solver_factor * self.jperp) * self.spin_kernel
        d0 = {(s1, s2): (solver_factor * self.szsz * (0.25 if s1 == s2 else -0.25)) * self.spin_kernel
              for s1 in SPINS for s2 in SPINS}
        return jperp_tau, d0

    def dyn_vertices_for_static_shift(self):
        """The Sz.Sz vertex list in `(a, b, D_tau)` form, for `static_shift`.

        Uses the CTHYB/CTSEG convention (`half_prefactor_action=True`), which is the one the
        solver actually applies, and includes every ordered pair as `static_shift` expects.
        """
        _, d0 = self.spin_couplings(half_prefactor_action=True)
        return [(SPIN_INDEX[s1], SPIN_INDEX[s2], d) for (s1, s2), d in d0.items()]

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
        """Particle-hole symmetric mu from the shipped helpers, including the K'(0) shift."""
        W_static = np.array([[0.0, self.U], [self.U, 0.0]])
        W_shift, level_shift = static_shift(self.dyn_vertices_for_static_shift(), 2, self.beta)
        mu = half_filling_mu(W_static, W_shift, level_shift)
        if not np.allclose(mu, mu[0]):
            raise AssertionError(f"mu should be spin-independent, got {mu}")
        return mu[0]

    def check_half_filling_mu(self, tol=1e-10):
        """Assert the helper reproduces U/2, and return the pieces so a run can report them."""
        W_shift, level_shift = static_shift(self.dyn_vertices_for_static_shift(), 2, self.beta)
        mu = self.half_filling_mu()
        if abs(mu - self.U / 2) > tol:
            raise AssertionError(
                f"Particle-hole symmetry requires mu = U/2 = {self.U / 2} for this model, but "
                f"half_filling_mu gave {mu}. W_shift =\n{W_shift}\nlevel_shift = {level_shift}")
        return dict(mu=mu, W_shift=W_shift, level_shift=level_shift)

    # ------------------------------------------------------------------ operators / bath

    def h_int(self):
        return self.U * n("up", 0) * n("down", 0)

    def h_loc0(self):
        return -self.mu * (n("up", 0) + n("down", 0))

    def delta_iw(self):
        """Hybridization on this model's Matsubara mesh, (1, 1) matrix_valued."""
        mesh = baths.imfreq_mesh(self.beta, self.n_iw)
        return baths.build_delta(mesh, self.args.bath, half_bandwidth=self.args.half_bandwidth,
                                 v_sq=self.args.V_sq, target_shape=(1, 1))

    def delta_iw_block(self):
        """`delta_iw` as a BlockGf over GF_STRUCT, which is what selfenergy.sigma_* expects."""
        from triqs.gfs import BlockGf
        delta = self.delta_iw()
        names = [name for name, _ in GF_STRUCT]
        return BlockGf(name_list=names, block_list=[delta.copy() for _ in names])

    def Q_tau_gf(self, target_shape=(1, 1)):
        return kernels.as_gf(self.Q, self.beta, target_shape=target_shape)

    # ------------------------------------------------------------------ bookkeeping

    def sigma_inputs(self):
        """`(mu, delta_iw_block)` -- the *input* mu and Delta, never read back from a solver
        whose `h_loc` has been mutated by the Lang-Firsov shift. See common/selfenergy.py."""
        return self.mu, self.delta_iw_block()

    def params(self):
        return dict(beta=self.beta, U=self.U, J=self.J, jperp=self.jperp, szsz=self.szsz,
                    mu=self.mu, filling=self.args.filling, bath=self.args.bath,
                    n_iw=self.n_iw, n_tau=self.n_tau, n_tau_bosonic=self.n_tau_bosonic)

    def output_file(self, solver, tag=""):
        return io.output_file(self.args.out_dir, BENCHMARK, solver, self.beta, self.args.filling,
                              tag=f"J-{self.J:g}_jperp-{self.jperp:g}_szsz-{self.szsz:g}"
                                  + (f"_{tag}" if tag else "")
                                  + (f"_seed-{self.args.seed}" if self.args.seed is not None else ""))

    def seed_kwargs(self):
        """`random_seed` for solve(), only when --seed was given (else the solver's default)."""
        return {} if self.args.seed is None else {"random_seed": self.args.seed}

    def report(self):
        """One-line summary every run prints, so a log says which model was solved."""
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
