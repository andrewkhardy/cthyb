# Copyright (c) 2025--present, The Simons Foundation
# Copyright (c) 2025--present, Max Planck Institute for Polymer Research, Mainz, Germany
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.

# Single orbital with dynamical spin-spin interactions.
# Data in spin_spin.ref.h5 is obtained by running this script on 800 cores.
import sys
import argparse
from triqs.gf import *
import triqs.utility.mpi as mpi
from triqs.gf.descriptors import Function
from triqs.operators import n
import h5
from triqs.utility.h5diff import h5diff
from triqs_cthyb import SolverCore as Solver
import matplotlib.pyplot as plt
from triqs.plot.mpl_interface import oplot

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run spin-spin benchmarking.')
parser.add_argument('--U', type=float, default=4.0, help='U parameter')
parser.add_argument('--J', type=float, default=1.0, help='J parameter')
parser.add_argument('--i1', type=float, default=1.0,  help='i1 switch (0 or 1)')
parser.add_argument('--i2', type=float, default=1.0,  help='i2 switch (0 or 1)')
parser.add_argument('--i3', type=float, default=1.0,  help='i3 switch (0 or 1)')
parser.add_argument('--i4', type=float, default=1.0,  help='i4 switch (0 or 1)')
parser.add_argument('--i5', type=float, default=1.0, help='i5 switch (0 or 1)')
args = parser.parse_args()
## for example python spin_spin.py --U 4.0 --J 2.0 --i1 1 --i2 1 --i3 1 --i4 1 --i5 1
# Numerical values
beta = 10
U = args.U
mu = U/2
J = args.J
i_1, i_2, i_3, i_4, i_5 = args.i1, args.i2, args.i3, args.i4, args.i5
n_tau = 4096
n_tau_bosonic = 2001

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
    Q_tau = Af["dmft_loop/i_000/Q_tau"]

# Hybridization Delta(tau)
n_iw = 1025
Delta = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
invg0 = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
invg0 << inverse(g0)
Delta << iOmega_n + mu - invg0
S.Delta_tau << Fourier(Delta)

# # Spin-spin interaction (D0(tau) and Jperp(tau))
# plt.figure()
# oplot(Q_tau)
# plt.show()
S.Jperp_tau << -(J**2) * Q_tau * i_1
S.D0_tau["up", "up"] << -0.25*J**2*Q_tau *i_2
S.D0_tau["down", "down"] << -0.25*J**2*Q_tau *i_3
S.D0_tau["up", "down"] << 0.25*J**2*Q_tau * i_4
S.D0_tau["down", "up"] << 0.25*J**2*Q_tau * i_5

# Solve parameters
solve_params = {
    "h_int": U*n("up", 0)*n("down", 0),
    "h_loc0": -mu * (n("up", 0) + n("down", 0)),
    "length_cycle": 100,
    "n_warmup_cycles": 100000,
    "n_cycles": 25000000,
    # Dynamical interaction moves are now automatically enabled when Jperp_tau or D0_tau are non-zero
    # "measure_F_tau": True,
    # "measure_nn_tau": True,
    # "measure_nn_static": True
    }

# Solve
S.solve(**solve_params)

# Save data
if mpi.is_master_node():
    filename = f"spin_spin_cthyb_J-{i_1}_U-{i_2}_{i_3}_{i_4}_{i_5}.h5"
    with h5.HDFArchive(filename, "w") as A:
        A['G_tau'] = S.G_tau
        # A['F_tau'] = S.F_tau
        # A['nn_tau'] = S.nn_tau
        # A['nn'] = S.nn_static
    print(f"Results saved to {filename}")
