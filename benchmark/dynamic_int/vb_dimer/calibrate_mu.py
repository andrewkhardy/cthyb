# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""Find the mu giving a target density, to pin in run_vb_dimer.sh (MU_B*_N* / MU_DCA_B*_N*).

  ED point   python calibrate_mu.py --target_n 0.75 --J_intra -0.5 --J_inter 0.5 --bath discrete --V 0.5
             exact ED probe (run_ed.py)
  DCA point  mpirun -n <N> python calibrate_mu.py --target_n 0.5 --J_intra 0 --J_inter 0.5 --bath dca
             no ED exists (-J indefinite), so a short CTHYB probe; needed at half filling too,
             since the dca bath breaks particle-hole symmetry
"""
import argparse
import os
import sys
from itertools import product

import numpy as np
import triqs.utility.mpi as mpi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import calibrate, selfenergy  # noqa: E402
import model as model_def  # noqa: E402
from model import N_PATCH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Calibrate mu for the vb_dimer benchmark')
model_def.add_model_args(parser)
parser.add_argument('--target_n', type=float, default=0.75, help='Target density per spin-orbital')
parser.add_argument('--probe', choices=['auto', 'ed', 'cthyb'], default='auto',
                    help="'auto': the ED whenever one exists for this coupling")
parser.add_argument('--tol', type=float, default=None,
                    help='Tolerance on the density; default 1e-6 for ED, 2e-3 for CTHYB')
parser.add_argument('--n_ph', type=int, default=2, help='ED phonon levels for the probe')
parser.add_argument('--probe_cycles', type=int, default=20000, help='CTHYB cycles per probe')
parser.add_argument('--length_cycle', type=int, default=50, help='CTHYB moves per cycle')
parser.add_argument('--dyn_n_l', type=int, default=50, help='Legendre coefficients for the dynamical interaction')
args = parser.parse_args()

# -J positive semidefinite is exactly the condition for a real-boson Hamiltonian, i.e. an ED.
J = np.array([[args.J_intra, args.J_inter], [args.J_inter, args.J_intra]])
ed_exists = args.bath == 'discrete' and np.linalg.eigvalsh(-J).min() > -1e-12
probe_name = args.probe if args.probe != 'auto' else ('ed' if ed_exists else 'cthyb')
if probe_name == 'ed' and not ed_exists:
    raise SystemExit(
        f"No ED for J_intra={args.J_intra}, J_inter={args.J_inter}, bath={args.bath}: -J has "
        f"eigenvalues {np.linalg.eigvalsh(-J)} and the bath must be 'discrete'. Use --probe cthyb.")
tol = args.tol if args.tol is not None else (1e-6 if probe_name == 'ed' else 2e-3)

BASE = ['--beta', str(args.beta), '--t', str(args.t), '--tp', str(args.tp), '--U', str(args.U),
        '--J_intra', str(args.J_intra), '--J_inter', str(args.J_inter),
        '--omega_0', str(args.omega_0), '--bath', args.bath, '--V', str(args.V),
        '--eps_bath', str(args.eps_bath), '--rotation', args.rotation,
        '--n_k', str(args.n_k), '--n_bins', str(args.n_bins)]


def density_cthyb(mu):
    """Mean density per spin-orbital from a short CTHYB run."""
    from triqs.gfs import Fourier
    from triqs_cthyb import Solver

    probe = argparse.Namespace(**vars(args))
    probe.mu = mu
    M = model_def.Model(probe)

    n_iw, n_tau, n_tau_bosonic = 256, 1025, 501
    S = Solver(beta=M.beta, gf_struct=M.gf_struct, n_iw=n_iw, n_tau=n_tau, n_l=30,
               n_tau_bosonic=n_tau_bosonic, delta_interface=True)
    for bl, delta in M.delta_iw(n_iw):
        S.Delta_tau[bl] << Fourier(delta)
    Q = M.Q(np.linspace(0, M.beta, n_tau_bosonic))
    for i, j in product(range(N_PATCH), repeat=2):
        D = -M.J[i, j] * Q
        S.add_dyn_int(D, M.Sz[i], M.Sz[j])
        S.add_dyn_int(D / 2, M.Sp[i], M.Sm[j])
        S.add_dyn_int(D / 2, M.Sm[i], M.Sp[j])

    # lang_firsov=True whatever the production setting: the one mu that the lf=True and lf=False
    # runs share must not depend on the routing.
    S.solve(h_int=M.h_int(), h_loc0=M.h_loc0(), lang_firsov=True,
            n_cycles=args.probe_cycles, n_warmup_cycles=max(args.probe_cycles // 20, 500),
            length_cycle=args.length_cycle, dyn_n_l=args.dyn_n_l,
            measure_G_tau=True, measure_G_l=True)

    # From G_l, not -G(beta): a short probe's -G(beta) is noisy enough to break the bisection.
    return float(np.mean(selfenergy.density_from_G_iw(selfenergy.G_iw_from_G_l(S.G_l, n_iw))))


guess = argparse.Namespace(**vars(args))
guess.mu = None
mu_half = float(model_def.Model(guess).mu)

if mpi.is_master_node():
    print(f"vb_dimer mu calibration ({probe_name} probe): beta={args.beta:g} U={args.U:g} "
          f"J_intra={args.J_intra:g} J_inter={args.J_inter:g} bath={args.bath}")
    print(f"  mu = {mu_half:.6f} is half filling"
          + ("" if model_def.Model(guess).half_filling_is_exact() else " only approximately "
             "(the coarse-grained bath breaks particle-hole symmetry)")
          + f"; scanning for n = {args.target_n}")

if probe_name == 'ed':
    density = calibrate.ed_probe(os.path.join(HERE, 'run_ed.py'), BASE + ['--n_ph', str(args.n_ph)])
else:
    density = density_cthyb
mu, n_achieved, samples = calibrate.bisect_mu(
    density, target_n=args.target_n, mu_guess=mu_half, tol=tol, step=0.5,
    verbose=mpi.is_master_node())

if mpi.is_master_node():
    variable = (f"MU_DCA_B{args.beta:g}" if args.bath == 'dca' else f"MU_B{args.beta:g}") \
        + f"_N{str(args.target_n).replace('.', '')}"
    print(calibrate.report(mu, n_achieved, args.target_n,
                           label=f"vb_dimer, beta = {args.beta:g}, J_intra = {args.J_intra:g}, "
                                 f"J_inter = {args.J_inter:g}, bath = {args.bath}, "
                                 f"probe = {probe_name}",
                           variable=variable))
