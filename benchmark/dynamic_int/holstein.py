# Copyright (c) 2025--present, The Simons Foundation
# Copyright (c) 2025--present, Max Planck Institute for Polymer Research, Mainz, Germany
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.

# Single orbital with dynamical spin-spin interactions.
# Data in spin_spin.ref.h5 is obtained by running this script on 800 cores.
import sys
import argparse
from triqs.gfs import *
import numpy as np
import triqs.utility.mpi as mpi
from triqs.gf.descriptors import Function
from triqs.gf.tools import *
from triqs.gf.block_gf import *
from triqs.operators import n
import h5
from triqs.utility.h5diff import h5diff
from triqs_cthyb import Solver
from triqs.operators import n
# Parse command line arguments
parser = argparse.ArgumentParser(description='Run spin-spin benchmarking.')
parser.add_argument('--U', type=float, default=4.0, help='U parameter')
parser.add_argument('--L', type=float, default=1.0, help='L parameter')
parser.add_argument('--beta', type=float, default=10.0, help='Inverse temperature')
parser.add_argument('--n_cycles', type=int, default=1000000, help='Number of MC cycles')
parser.add_argument('--measure_O_tau', type=int, default=100, help='Minimum insertions for O_tau measurement')
args, unknown = parser.parse_known_args()

# Numerical values
hopping = 1.0
w0 = 0.1
beta = args.beta
U = args.U
L = args.L
n_tau = 4096
n_tau_bosonic = 3999
n_iw = 1024
n_cycles = args.n_cycles
measure_O_tau_min_ins = args.measure_O_tau
Sz = 0.5 * ( n('up', 0) - n('down', 0) )
gf_struct = [('down', 1), ('up', 1)]
constr_params = {
    "gf_struct": gf_struct,
    "beta": beta,
    "n_tau": n_tau,
    "n_tau_bosonic": n_tau_bosonic,
    "delta_interface": True  # Use Delta_tau interface for dynamical interactions
}

# Construct solver
S = Solver(**constr_params)

# Hybridization Delta(tau)
Delta = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
g0 = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
q_tau = GfImTime(indices=[0],  statistic='Boson', beta=beta, n_points=n_tau_bosonic)
q_iw = make_gf_from_fourier(q_tau)  
q_iw << Function(lambda w: 2 * L/w0 * w0**2 / (w**2 - w0**2))
q_tau << Fourier(q_iw)
Q_tau = Block2Gf(['up', 'down'], ['up', 'down'], [[q_tau, q_tau], [q_tau, q_tau]])
Q_iw = make_gf_from_fourier(Q_tau)
g0 << SemiCircular(2*hopping)
ivn = np.array([x.imag for x in Q_iw["up", "up"].mesh.values()])
zero_freq = np.where(np.abs(ivn) < 1e-10)
mu = U/2 + np.real((Q_iw["up", "up"].data[zero_freq][0,0,0]+Q_iw["up", "down"].data[zero_freq][0,0,0]))/2.0
Delta << hopping**2*SemiCircular(2*hopping)
S.Delta_tau << Fourier(Delta)

S.D0_tau << Q_tau


# Solve parameters
Sz = 0.5 * ( n('up', 0) - n('down', 0) )
solve_params = {
    "h_int": U*n("up", 0)*n("down", 0),
    "h_loc0": -mu * (n("up", 0) + n("down", 0)),
    "length_cycle": 100,
    "n_warmup_cycles": 10000,
    "n_cycles": n_cycles,
    "measure_pert_order": True,
    "measure_O_tau": (Sz, Sz),
    "measure_O_tau_min_ins": measure_O_tau_min_ins,
    "perform_tail_fit": True,
    "fit_max_moment": 3,
    "lang_firsov": True
    }

# Solve
S.solve(**solve_params)

# Save data
if mpi.is_master_node():
    filename = f"/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin_cthyb_lambda-{L}-U-{U}_b-{beta}_nw-{solve_params['n_cycles']}_mins-{solve_params['measure_O_tau_min_ins']}.h5"
    with h5.HDFArchive(filename, "w") as A:
        A['G_tau'] = S.G_tau
        A["perturbation_order"] = S.perturbation_order
        A["average_sign"] = S.average_sign
        A["O_tau"] = S.O_tau#[(Sz, Sz)], hopefully allows many measurements eventually? # why use this over G2 blocks? 
        A["Sigma_iw"] = S.Sigma_iw
        A["Sigma_tau"] = S.Sigma_tau
        # A['F_tau'] = S.F_tau
        # A['nn_tau'] = S.nn_tau
        # A['nn'] = S.nn_static
    print(f"Results saved to {filename}")
