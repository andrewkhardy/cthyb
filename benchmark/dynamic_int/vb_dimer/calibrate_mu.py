# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""Find the mu giving a target density, to pin in run_vb_dimer.sh so every run of a given
point solves the identical Hamiltonian.

Two points, two probes, because only one of them has an ED:

  ED point   python calibrate_mu.py --target_n 0.75 \
                    --J_intra -0.5 --J_inter 0.5 --bath discrete --V 0.5
             Exact: run_ed.py carries no statistical error, so the bisection is limited
             only by the phonon truncation. Submit it rather than running it locally --
             each solve is a dense eigh on a ~2000-4500 dimensional block.

  DCA point  mpirun -n <N> python calibrate_mu.py --target_n 0.5 \
                    --J_intra 0 --J_inter 0.5 --bath dca
             -J is indefinite here, so no ED exists (run_ed.py refuses it by design) and a
             short low-statistics CTHYB run is the only probe. Note the DCA point needs
             calibrating at HALF FILLING too: the coarse-grained hybridization is not
             particle-hole symmetric (model.half_filling_is_exact() is False), so mu = U/2
             is only approximate there -- unlike --bath discrete, where it is exact.

Paste the printed value into MU_B10_N075 / MU_B100_N075.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import triqs.utility.mpi as mpi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import calibrate, selfenergy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as model_def  # noqa: E402

parser = argparse.ArgumentParser(description='Calibrate mu for the vb_dimer benchmark')
model_def.add_model_args(parser)
parser.add_argument('--target_n', type=float, default=0.75,
                    help='Target density per spin-orbital (0.5 is half filling)')
parser.add_argument('--probe', choices=['auto', 'ed', 'cthyb'], default='auto',
                    help="'auto' uses the ED whenever one exists for this coupling")
parser.add_argument('--tol', type=float, default=None,
                    help='Tolerance on the density; default 1e-6 for ED, 2e-3 for CTHYB')
parser.add_argument('--n_ph', type=int, default=2, help='ED phonon levels for the probe')
parser.add_argument('--probe_cycles', type=int, default=20000,
                    help='CTHYB cycles per probe point. Keep small: the bisection cannot '
                         'resolve mu below the probe noise, so more cycles only cost time')
parser.add_argument('--length_cycle', type=int, default=50, help='CTHYB moves per cycle')
parser.add_argument('--dyn_n_l', type=int, default=50,
                    help='Legendre coefficients for the dynamical interaction')
args = parser.parse_args()

# -J psd is exactly the condition for a real-boson Hamiltonian, i.e. for an ED to exist.
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


def density_ed(mu):
    """Mean density per spin-orbital from one exact ED solve.

    run_ed.py builds its Hamiltonian at import time from argparse, so it is a script, not a
    library: drive it as a subprocess and read the <n_a> line it already prints.
    """
    out = subprocess.run(
        [sys.executable, os.path.join(HERE, 'run_ed.py'), *BASE, '--mu', repr(float(mu)),
         '--n_ph', str(args.n_ph), '--n_ph_check', '0', '--n_tau', '101', '--n_iw', '64',
         '--out_dir', os.path.join(HERE, 'data', 'calib')],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"run_ed.py failed at mu={mu}:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    for line in out.stdout.splitlines():
        if line.startswith('<n_a> ='):
            vals = line.split('=', 1)[1].split('(')[0].strip().strip('[]').split()
            return float(np.mean([float(v) for v in vals]))
    raise RuntimeError(f"could not parse <n_a> from run_ed.py output:\n{out.stdout[-2000:]}")


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
    M.register_vertices(S, n_tau_bosonic, basis='site')

    # lang_firsov=True unconditionally, whatever the production setting. The probe defines
    # the ONE mu that both production runs share; letting it follow --lang_firsov would give
    # the lf=True and lf=False runs different Hamiltonians and destroy the only cross-check
    # available at the DCA point. It is also the cheaper, lower-variance route.
    S.solve(h_int=M.h_int(), h_loc0=M.h_loc0(), lang_firsov=True,
            n_cycles=args.probe_cycles, n_warmup_cycles=max(args.probe_cycles // 20, 500),
            length_cycle=args.length_cycle, dyn_n_l=args.dyn_n_l,
            measure_G_tau=True, measure_G_l=True)

    # From G_l -> G(iw).density(), never from -G_tau.data[-1]: a 20k-cycle -G(beta) scatters
    # enough to make n(mu) non-monotonic, which breaks the bisection outright.
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

density = density_ed if probe_name == 'ed' else density_cthyb
mu, n_achieved, samples = calibrate.bisect_mu(
    density, target_n=args.target_n, mu_guess=mu_half, tol=tol, step=0.5,
    verbose=mpi.is_master_node())

if mpi.is_master_node():
    mu_fit = calibrate.interpolate_from_samples(samples, args.target_n)
    variable = (f"MU_DCA_B{args.beta:g}" if args.bath == 'dca' else f"MU_B{args.beta:g}") \
        + f"_N{str(args.target_n).replace('.', '')}"
    print(calibrate.report(mu_fit, n_achieved, args.target_n,
                           label=f"vb_dimer, beta = {args.beta:g}, J_intra = {args.J_intra:g}, "
                                 f"J_inter = {args.J_inter:g}, bath = {args.bath}, "
                                 f"probe = {probe_name}",
                           variable=variable))
    print(f"  (bisection endpoint {mu:.6f}, linear fit over {len(samples)} probes {mu_fit:.6f})")
