# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# Multi-orbital (2-orbital) impurity with dynamical spin-spin and density-density interactions.
# This is the cthyb analog of the ctseg dynamic_int_multiorb example,
# but uses Jperp_tau (spin-flip dynamical interaction) which ctseg cannot handle.
#
# The structure closely follows spin_spin.py but generalizes to 2 orbitals.
#
# gf_struct = [('up_0', 1), ('down_0', 1), ('up_1', 1), ('down_1', 1)]
#   i.e. each orbital+spin is its own block, interleaved (up_a, down_a) pairs
#   (as required by cthyb for D0_tau indexing and Jperp interleaved mode).
#
# Dynamical interactions:
#   D0_tau[s1_a, s2_b] : density-density retarded interaction between (spin s1, orb a) and (spin s2, orb b)
#   Jperp_tau           : spin-flip (S+S-) retarded interaction (scalar, applied globally)

import sys
import argparse
import numpy as np
from triqs.gf import *
import triqs.utility.mpi as mpi
from triqs.gf.descriptors import Function
from triqs.operators import n, c, c_dag, Operator
import h5
from triqs_cthyb import SolverCore as Solver

# ======================== Command line arguments (falls back to defaults in interactive window) ========================
parser = argparse.ArgumentParser(description='Multi-orbital cthyb benchmark with dynamical spin-spin interactions.')
parser.add_argument('--U',    type=float, default=4.0,  help='Intra-orbital Hubbard U')
parser.add_argument('--Up',   type=float, default=0.0,  help='Inter-orbital Hubbard U\'')
parser.add_argument('--J',    type=float, default=1.0,  help='Dynamical interaction coupling J')
parser.add_argument('--beta', type=float, default=10.0, help='Inverse temperature')
parser.add_argument('--t',    type=float, default=1.0,  help='Hopping (half-bandwidth for Bethe lattice)')
parser.add_argument('--n_cycles', type=int, default=500000, help='Number of MC cycles')
parser.add_argument('--length_cycle', type=int, default=100, help='Length of MC cycle')
parser.add_argument('--n_warmup_cycles', type=int, default=50000, help='Warmup cycles')
try:
    args = parser.parse_args()
except SystemExit:
    args = parser.parse_args([])

# ======================== Parameters ========================
beta = args.beta
U    = args.U
Up   = args.Up
J_dyn = args.J
t    = args.t       # half-bandwidth
mu   = (U + Up) / 2  # half-filling for 2-orbital model with U, U'
n_orb = 2
n_tau = 10001
n_tau_bosonic = 10001
n_iw  = 1025


# Block structure: each (spin, orbital) is a separate block (interleaved: up_0, down_0, up_1, down_1, ...)
# This is required so that D0_tau can be indexed as D0_tau["up_0", "down_1"] etc.
spins = ['up', 'down']
gf_struct = [('%s_%i' % (s, o), 1) for o in range(n_orb) for s in spins]

#if mpi.is_master_node():
# print("=" * 60)
# print("Multi-orbital cthyb with dynamical interactions")
# print("=" * 60)
# print(f"  n_orb       = {n_orb}")
# print(f"  U           = {U}")
# print(f"  U'          = {Up}")
# print(f"  J_dyn       = {J_dyn}")
# print(f"  Jperp_on    = {Jperp_on}")
# print(f"  D0_on       = {D0_on}")
# print(f"  beta        = {beta}")
# print(f"  t (half-bw) = {t}")
# print(f"  mu          = {mu}")
# print(f"  gf_struct   = {gf_struct}")
# print("=" * 60)

# ======================== Construct solver ========================
constr_params = {
    "gf_struct":       gf_struct,
    "beta":            beta,
    "n_tau":           n_tau,
    "n_tau_bosonic":   n_tau_bosonic,
    "delta_interface": True,
}

S = Solver(**constr_params)

# ======================== Hybridization: Bethe lattice ========================
# Bethe lattice self-consistency: Delta(iw) = t^2 * G_loc(iw)
# At the non-interacting starting point (Sigma=0, half-filling):
#   G_loc = G_semicircular with half-bandwidth t
# Using TRIQS SemiCircular descriptor ensures proper tail handling for Fourier.

from triqs.gf.descriptors import SemiCircular

for s in spins:
    for o in range(n_orb):
        block_name = '%s_%i' % (s, o)
        G_loc = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
        G_loc << SemiCircular(t)
        Delta_iw = G_loc.copy()
        Delta_iw << t**2 * G_loc
        S.Delta_tau[block_name] << Fourier(Delta_iw)

# ======================== Dynamical interaction Q(tau) ========================
# We use a simple model for the retarded interaction kernel Q(tau).
# A common choice is Q(tau) = -lambda * exp(-omega_0 * tau*(beta-tau)/beta)
# or from a single bosonic mode: Q(tau) ~ cosh(omega_0 * (tau - beta/2)) / sinh(omega_0 * beta/2)
#
# For simplicity, use a single bosonic mode with frequency omega_0:
#   Q(tau) = -(1/(2*omega_0)) * cosh(omega_0*(tau - beta/2)) / sinh(omega_0*beta/2)
# This is the standard retarded interaction from integrating out a single bosonic mode.

omega_0 = 1.0  # bosonic mode frequency

tau_mesh_pts = np.linspace(0, beta, n_tau_bosonic)
Q_data = np.zeros(n_tau_bosonic)
for i, tau in enumerate(tau_mesh_pts):
    Q_data[i] = -(1.0 / (2.0 * omega_0)) * np.cosh(omega_0 * (tau - beta / 2.0)) / np.sinh(omega_0 * beta / 2.0)

Q_tau = GfImTime(indices=[0], beta=beta, statistic='Boson', n_points=n_tau_bosonic)
Q_tau.data[:, 0, 0] = Q_data

# if mpi.is_master_node():
# print(f"Q(0)    = {Q_data[0]:.6f}")
# print(f"Q(beta) = {Q_data[-1]:.6f}")
# print(f"Q(beta/2) = {Q_data[n_tau_bosonic//2]:.6f}")

# ======================== Set dynamical interactions ========================
# Jperp_tau: spin-flip dynamical interaction, now a Block2Gf indexed by (block1, block2).
# For the interleaved layout, Jperp for orbital pair (a, b) is stored in
# Jperp_tau["up_a", "up_b"]. The C++ code reads from the up-up block pairs.
# SU(2) symmetric: same Jperp for all orbital pairs.
for a in range(n_orb):
        S.Jperp_tau["up_%i" % a, "up_%i" % a] << -(J_dyn) * Q_tau

# D0_tau: density-density dynamical interaction
# D0_tau is a Block2Gf indexed by (block_name_1, block_name_2)
# Following the spin_spin.py convention for the Sz*Sz decomposition:
#   D0["up_a", "up_b"]   << -0.25 * J * Q_tau   (same spin: -1/4)
#   D0["down_a", "down_b"] << -0.25 * J * Q_tau (same spin: -1/4)
#   D0["up_a", "down_b"] << +0.25 * J * Q_tau   (opposite spin: +1/4)
#   D0["down_a", "up_b"] << +0.25 * J * Q_tau   (opposite spin: +1/4)
#
# This corresponds to the Sz*Sz part of the spin-spin interaction:
#   J * Sz_a * Sz_b * Q(tau) = J * (1/4)(n_up_a - n_down_a)(n_up_b - n_down_b) * Q(tau)
# expanded:  (1/4)[n_up_a*n_up_b - n_up_a*n_down_b - n_down_a*n_up_b + n_down_a*n_down_b]
# so same-spin gets -J/4*Q and opposite-spin gets +J/4*Q
#
# We apply this for all orbital pairs (including intra-orbital).
for a in range(n_orb):
    # Same spin: -0.25 * J
    S.D0_tau["up_%i" % a, "up_%i" % a] << -0.25 * J_dyn * Q_tau
    S.D0_tau["down_%i" % a, "down_%i" % a] << -0.25 * J_dyn * Q_tau
    # Opposite spin: +0.25 * J
    S.D0_tau["up_%i" % a, "down_%i" % a] << 0.25 * J_dyn * Q_tau
    S.D0_tau["down_%i" % a, "up_%i" % a] << 0.25 * J_dyn * Q_tau

# ======================== Static Hamiltonian ========================
# h_int: interacting part (quartic terms)
# Intra-orbital: U * n_up_a * n_down_a
# Inter-orbital: U' * n_a * n_b  (for a != b, simplified)
h_int = Operator()
for a in range(n_orb):
    h_int += U * n('up_%i' % a, 0) * n('down_%i' % a, 0) 
for a in range(n_orb):
    for b in range(n_orb):
        if a != b:
            for s1 in spins:
                for s2 in spins:
                    h_int += (Up / 2.0) * n('%s_%i' % (s1, a), 0) * n('%s_%i' % (s2, b), 0)
                    
# more Hund's interactions here?
# h_loc0: quadratic part (chemical potential)
h_loc0 = Operator()
for s in spins:
    for a in range(n_orb):
        h_loc0 += -mu * n('%s_%i' % (s, a), 0)
        # need to add DCA buisness here. 

print("h_int =", h_int)
print("h_loc0 =", h_loc0)

# ======================== Solve parameters ========================
solve_params = {
    "h_int":             h_int,
    "h_loc0":            h_loc0,
    "length_cycle":      args.length_cycle,
    "n_warmup_cycles":   args.n_warmup_cycles,
    "n_cycles":          args.n_cycles,
    "measure_pert_order": True,
    "measure_G_tau":     True,
}

# ======================== Solve ========================
S.solve(**solve_params)

# ======================== Save results ========================

filename = (f"multiorb_spin_spin_cthyb_norb-{n_orb}_U-{U}_Up-{Up}_J-{J_dyn}"
            f"_{U}_{J_dyn}_beta-{beta}_nc-{args.n_cycles}.h5")
with h5.HDFArchive(filename, "w") as A:
    A['G_tau'] = S.G_tau
    A['perturbation_order'] = S.perturbation_order
    A['average_sign'] = S.average_sign
    A["perturbation_order_dynamical"] = S.perturbation_order_dyn
    # Save parameters for reproducibility
    A['U']     = U
    A['Up']    = Up
    A['J_dyn'] = J_dyn
    A['beta']  = beta
    A['n_orb'] = n_orb
    A['Q_tau']    = Q_tau
print(f"Results saved to {filename}")
print(f"Average sign = {S.average_sign}")
