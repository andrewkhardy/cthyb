# Copyright (c) 2025--present, The Simons Foundation
# Copyright (c) 2025--present, Max Planck Institute for Polymer Research, Mainz, Germany
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.

# Single orbital with dynamical spin-spin interactions.
# Data in spin_spin.ref.h5 is obtained by running this script on 800 cores.
import sys
import numpy as np
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
# Parse command line arguments
parser = argparse.ArgumentParser(description='Run spin-spin benchmarking.')
parser.add_argument('--U', type=float, default=4.0, help='U parameter')
parser.add_argument('--J', type=float, default=1.0, help='J parameter')
parser.add_argument('--i1', type=float, default=1.0,  help='i1 switch (0 or 1)')
parser.add_argument('--i2', type=float, default=1.0,  help='i2 switch (0 or 1)')
parser.add_argument('--i3', type=float, default=1.0,  help='i3 switch (0 or 1)')
parser.add_argument('--i4', type=float, default=1.0,  help='i4 switch (0 or 1)')
parser.add_argument('--i5', type=float, default=1.0,  help='i5 switch (0 or 1)')
parser.add_argument('--beta', type=float, default=10.0, help='Inverse temperature')
parser.add_argument('--n_cycles', type=int, default=1000000, help='Number of MC cycles')
parser.add_argument('--measure_O_tau', type=int, default=100, help='Minimum insertions for O_tau measurement')
parser.add_argument('--dyn_n_l', type=int, default=100, help='Number of Legendre polynomials for dynamical interactions')
args, unknown = parser.parse_known_args()
# Numerical values
beta = args.beta
dyn_n_l = args.dyn_n_l
U = args.U
mu = U/2
J = args.J
i_1, i_2, i_3, i_4, i_5 = args.i1, args.i2, args.i3, args.i4, args.i5
n_tau = 4096
n_tau_bosonic = 2001
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

# Get inputs from reference file
with h5.HDFArchive("ctint.ref.h5", 'r') as Af:
    g0 = Af["dmft_loop/i_001/S/G0_iw/up"]
    q_tau = Af["dmft_loop/i_000/Q_tau"]



# Hybridization Delta(tau)
n_iw = len(g0.mesh)
Delta = GfImFreq(indices=[0], beta=beta, n_points=n_iw//2)
invg0 = GfImFreq(indices=[0], beta=beta, n_points=n_iw//2)
Q_tau = GfImTime(indices=[0], statistic='Boson', beta=beta, n_points=n_tau_bosonic)

G0 = invg0.copy()
G0.data[:,0,0] = g0.data[:,0,0]
Q_tau.data[:,0,0] = q_tau.data[:,0,0]
invg0 << inverse(g0)
Delta << iOmega_n + mu - invg0
S.Delta_tau << Fourier(Delta)

# Shift Q_tau so its integral over beta is exactly zero
q_tau_mean = np.mean(Q_tau.data[:, 0, 0])
print(q_tau_mean)
print("Shifting Q_tau by its mean value to ensure zero integral over beta.")
#Q_tau.data[:, 0, 0] = Q_tau.data[:, 0, 0] - q_tau_mean

# Spin-spin interaction (D0(tau) and Jperp(tau))
S.Jperp_tau << -(J) * Q_tau * i_1
S.D0_tau["up", "up"] << -0.25*J*Q_tau * i_2
S.D0_tau["down", "down"] << -0.25*J*Q_tau * i_3
S.D0_tau["up", "down"] << 0.25*J*Q_tau * i_4
S.D0_tau["down", "up"] << 0.25*J*Q_tau * i_5

# Solve parameters
Sz = 0.5 * ( n('up', 0) - n('down', 0) )
solve_params = {
    "h_int": U*n("up", 0)*n("down", 0),
    "h_loc0": -mu * (n("up", 0) + n("down", 0)),
    "length_cycle": 100,
    "n_warmup_cycles": 50000,
    "n_cycles": n_cycles,
    "measure_pert_order": True,
    "measure_O_tau": (Sz, Sz),
    "measure_O_tau_min_ins": measure_O_tau_min_ins,
    # "perform_tail_fit": True,
    # "fit_max_moment": 3,
    "lang_firsov": True,
    "dyn_n_l": dyn_n_l
    }

# Solve
S.solve(**solve_params)

# Save data
if mpi.is_master_node():
    #filename = f"/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin_cthyb_J-{J}-U-{U}_{i_1}_{i_2}_{i_3}_{i_4}_{i_5}_b-{beta}_nw-{solve_params['n_cycles']}_mins-{solve_params['measure_O_tau_min_ins']}_lf={solve_params['lang_firsov']}.h5"
    filename = f"/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin_cthyb_J-{J}-U-{U}_{i_1}_b-{beta}_nw-{solve_params['n_cycles']}_mins-{solve_params['measure_O_tau_min_ins']}_lf={solve_params['lang_firsov']}_nl={solve_params['dyn_n_l']}.h5"

    with h5.HDFArchive(filename, "w") as A:
        A['G_tau'] = S.G_tau
        A["perturbation_order"] = S.perturbation_order
        A["average_sign"] = S.average_sign
        A["O_tau"] = S.O_tau#[(Sz, Sz)], hopefully allows many measurements eventually? # why use this over G2 blocks? 
        A["Sigma_iw"] = S.Sigma_iw
        #A["Sigma_tau"] = S.Sigma_tau
        # A['F_tau'] = S.F_tau
        # A['nn_tau'] = S.nn_tau
        # A['nn'] = S.nn_static
    print(f"Results saved to {filename}")
