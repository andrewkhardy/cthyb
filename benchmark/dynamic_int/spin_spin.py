# Copyright (c) 2025--present, The Simons Foundation
# Copyright (c) 2025--present, Max Planck Institute for Polymer Research, Mainz, Germany
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.

# Single orbital with dynamical spin-spin interactions.
# Data in spin_spin.ref.h5 is obtained by running this script on 800 cores.
import sys
sys.path.insert(0, '/home/andrewhardy/Documents/CCQ/cthyb_dyn/build/python')
from triqs.gf import *
import triqs.utility.mpi as mpi
from triqs.gf.descriptors import Function
from triqs.operators import n
import h5
from triqs.utility.h5diff import h5diff
from triqs_cthyb import SolverCore as Solver

# Numerical values
beta = 10
U = 4.0
mu = U/2
J = 0.5*2
n_tau = 4096
n_tau_bosonic = 2001

# Solver construction parameters
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

# Spin-spin interaction (D0(tau) and Jperp(tau))
S.Jperp_tau << -(J**2) * Q_tau *0
S.D0_tau["up", "up"] << -0.25*J**2*Q_tau
S.D0_tau["down", "down"] << -0.25*J**2*Q_tau
S.D0_tau["up", "down"] << 0.25*J**2*Q_tau/2
S.D0_tau["down", "up"] << 0.25*J**2*Q_tau/2

# Solve parameters
solve_params = {
    "h_int": U*n("up", 0)*n("down", 0),
    "h_loc0": -mu * (n("up", 0) + n("down", 0)),
    "length_cycle": 50,
    "n_warmup_cycles": 50000,
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
    with h5.HDFArchive("spin_spin_cthyb_nhalf-1.h5", "w") as A:
        A['G_tau'] = S.G_tau
        # A['F_tau'] = S.F_tau
        # A['nn_tau'] = S.nn_tau
        # A['nn'] = S.nn_static
    print("Results saved to spin_spin_cthyb_nhalf-1.h5")
