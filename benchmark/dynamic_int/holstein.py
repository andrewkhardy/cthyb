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
from triqs.gfs.descriptors import Function
from triqs.gfs.tools import *
from triqs.gfs.block_gf import *
from triqs.operators import n
import h5
from triqs.utility.h5diff import h5diff
from triqs_cthyb import Solver
from triqs.atom_diag import trace_rho_op
# Parse command line arguments
parser = argparse.ArgumentParser(description='Run spin-spin benchmarking.')
parser.add_argument('--L', type=float, default=1.0, help='L parameter')
parser.add_argument('--U', type=float, default=4.0, help='U parameter')
parser.add_argument('--beta', type=float, default=10.0, help='Inverse temperature')
parser.add_argument('--n_cycles', type=int, default=1000000, help='Number of MC cycles')
parser.add_argument('--measure_O_tau', type=int, default=100, help='Minimum insertions for O_tau measurement')
parser.add_argument("--dyn_n_l", type=int, default=100, help="Number of Legendre polynomials for dynamical interactions")
parser.add_argument("--lang_firsov", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False, help="Whether to use Lang-Firsov approach for dynamical interactions")
args, unknown = parser.parse_known_args()
# Numerical values
hopping = 1.0
w0 = 0.1
beta = args.beta
lang_firsov = args.lang_firsov
dyn_n_l = args.dyn_n_l
U = args.U
L = args.L
n_tau = 4096*2
n_tau_bosonic = 4096*2
n_iw = 1025
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
g0 = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
Delta = g0.copy()
q_tau = GfImTime(indices=[0],  statistic='Boson', beta=beta, n_points=n_tau_bosonic)
q_iw = make_gf_from_fourier(q_tau)  

q_iw << Function(lambda w: 2 * L/w0 * w0**2 / (w**2 - w0**2))

q_tau << Fourier(q_iw)
q_tau_mean = np.mean(q_tau.data[:, 0, 0])
print(q_tau_mean, "q_tau_mean")

Q_tau = Block2Gf(['up', 'down'], ['up', 'down'], [[1*q_tau, 1*q_tau], [1*q_tau, 1*q_tau]])
Q_iw = make_gf_from_fourier(Q_tau)
g0 << SemiCircular(2*hopping)
ivn = np.array([x.imag for x in Q_iw["up", "up"].mesh.values()])
zero_freq = np.where(np.abs(ivn) < 1e-10)
mu = U/2 + np.real((Q_iw["up", "up"].data[zero_freq][0,0,0]+Q_iw["up", "down"].data[zero_freq][0,0,0]))/2.0
Delta << SemiCircular(2*hopping)
S.Delta_tau << Fourier(Delta)

S.D0_tau << Q_tau
# Solve parameters
Sz = 0.5 * ( n('up', 0) - n('down', 0) )
solve_params = {
    "h_int": U*n("up", 0)*n("down", 0),
    "h_loc0": -mu * (n("up", 0) + n("down", 0)),
    "length_cycle": 100,
    "n_warmup_cycles": 40000,
    "n_cycles": n_cycles,
    "measure_pert_order": True,
    "measure_O_tau": (Sz, Sz),
    "measure_O_tau_min_ins": measure_O_tau_min_ins,
    # "perform_tail_fit": True,
    # "fit_max_moment": 3,
    "dyn_n_l": dyn_n_l,
    "lang_firsov": lang_firsov,
    "measure_D0_corr": True,
    "measure_density_matrix": True,
    "use_norm_as_weight": True,
    }

# Solve
S.solve(**solve_params)

# Save data
if mpi.is_master_node():
    # Compute SzSz constant offset from density matrix
    rho = S.density_matrix
    h_loc_diag = S.h_loc_diagonalization
    n_up_val = trace_rho_op(rho, n("up", 0), h_loc_diag).real
    n_dn_val = trace_rho_op(rho, n("down", 0), h_loc_diag).real
    double_occ = trace_rho_op(rho, n("up", 0) * n("down", 0), h_loc_diag).real
    SzSz_offset = 0.25 * (n_up_val + n_dn_val - 2.0 * double_occ)
    print(f"n_up = {n_up_val:.6f}")
    print(f"n_down = {n_dn_val:.6f}")
    print(f"<n_up*n_down> = {double_occ:.6f}")
    print(f"SzSz_offset = <Sz^2> = {SzSz_offset:.6f}")

    filename = f"/mnt/home/ahardy/ceph/CTHYB_Data/cthyb_lambda-{L}-U-{U}_b-{beta}_nw-{solve_params['n_cycles']}_mins-{solve_params['measure_O_tau_min_ins']}_lf={lang_firsov}.h5"
    with h5.HDFArchive(filename, "w") as A:
        A['G_tau'] = S.G_tau
        A["perturbation_order"] = S.perturbation_order
        A["average_sign"] = S.average_sign
        A["O_tau"] = S.O_tau
        A["Sigma_iw"] = S.Sigma_iw
        A["K_n"] = S.K_n
        A["Q_tau"] = S.Q_tau
        A["Q_l"] = S.Q_l
        A["n_up"] = n_up_val
        A["n_down"] = n_dn_val
        A["double_occ"] = double_occ
        A["SzSz_offset"] = SzSz_offset
    print(f"Results saved to {filename}")
