# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# <S^+(tau) S^-(0)> with a Holstein phonon on n_up alone, against exact diagonalization of the impurity, one bath site
# per spin and the phonon (truncated at N_PH quanta).
#
# The phonon enters CTHYB as the retarded n_up n_up interaction, resummed by Lang-Firsov. S^+ = c^dagger_up c_do moves an
# electron out of the density the phonon couples to, so its kinks change the Lang-Firsov weight of every configuration:
# the measurement has to include that factor, the polaron dressing of the spin flip. Without it the correlator was off
# by 0.06 here, eleven times the noise.

from functools import reduce
import numpy as np
from triqs.operators import n, c, c_dag
from triqs_cthyb import Solver

beta, U, mu, V, eps = 5.0, 2.0, 0.7, 0.8, 0.3
omega_0, g = 1.0, 1.2
N_PH, n_tau = 30, 1001


def Q(tau):
    """The phonon propagator of model.py in benchmark/dynamic_int/kanamori_phonon: X (d + d^dagger) / sqrt(2 omega_0)"""
    return -np.cosh(omega_0 * (tau - beta / 2)) / (2 * omega_0 * np.sinh(omega_0 * beta / 2))


def exact(taus):
    """<S^+(tau) S^-(0)> by exact diagonalization, fermions by Jordan-Wigner times the phonon's Fock space"""
    n_modes = 4  # 0, 1: impurity up, do; 2, 3: bath up, do
    a, z, one = np.array([[0.0, 1.0], [0.0, 0.0]]), np.diag([1.0, -1.0]), np.eye(2)
    cf = [reduce(np.kron, [z] * j + [a] + [one] * (n_modes - j - 1)) for j in range(n_modes)]
    d = np.diag(np.sqrt(np.arange(1, N_PH)), 1)
    I_f, I_b = np.eye(2**n_modes), np.eye(N_PH)
    c_ = [np.kron(x, I_b) for x in cf]
    num = [x.T @ x for x in c_]
    H = U * num[0] @ num[1] - mu * (num[0] + num[1])
    for s in range(2):
        H += eps * num[2 + s] + V * (c_[s].T @ c_[2 + s] + c_[2 + s].T @ c_[s])
    H += omega_0 * np.kron(I_f, d.T @ d) + g * num[0] @ np.kron(I_f, d + d.T) / np.sqrt(2 * omega_0)
    E, W = np.linalg.eigh(H)
    E -= E.min()
    S_plus = W.T @ (c_[0].T @ c_[1]) @ W
    Z = np.exp(-beta * E).sum()
    return np.array([np.exp(-(beta - t) * E) @ (S_plus * S_plus) @ np.exp(-t * E) for t in taus]) / Z


S = Solver(beta=beta, gf_struct=[('up', 1), ('do', 1)], n_iw=200, n_tau=10001, n_tau_bosonic=n_tau, delta_interface=True)
for _, delta in S.Delta_tau:
    delta.data[:, 0, 0] = [-V**2 * np.exp(-eps * float(t)) / (1 + np.exp(-beta * eps)) for t in delta.mesh]
S.add_dyn_int(g**2 * Q(np.linspace(0, beta, n_tau)), n('up', 0), n('up', 0))

S_plus, S_minus = c_dag('up', 0) * c('do', 0), c_dag('do', 0) * c('up', 0)
S.solve(h_int=U * n('up', 0) * n('do', 0), h_loc0=-mu * (n('up', 0) + n('do', 0)), n_cycles=100000, n_warmup_cycles=5000,
        length_cycle=20, random_seed=7121, measure_G_tau=False, measure_O_tau=(S_minus, S_plus), verbosity=0)

taus = np.linspace(0, beta, 51)
qmc, ref = np.array([S.O_dlr(t).real for t in taus]), exact(taus)
print(f"max |QMC - ED| of <S+(tau) S-(0)>: {np.abs(qmc - ref).max():.4f}  (correlator up to {ref.max():.3f})")
np.testing.assert_allclose(qmc, ref, atol=0.02)
