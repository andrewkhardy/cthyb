# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# The Pauli proposal of the pair moves (pauli_prob) and the default of move_double, against exact diagonalization of the
# impurity and one bath site per orbital:
#   1. Blocks of size 1 and a density-density h_loc, where by default pauli_prob = 1 and the four-operator moves are off:
#      G agrees with ED for pauli_prob = 0, 0.5 and 1.
#   2. One block of size 2 with an inter-orbital hopping in h_loc, so that the operators of an index need not alternate and
#      the uniform part of the proposal is needed: G agrees with ED for pauli_prob = 0.5.
# G is compared in its Legendre representation, the ED one truncated at the same n_l. These are statistical checks with
# fixed seeds: the tolerance is several times the Monte Carlo noise.

from functools import reduce
import numpy as np
from numpy.polynomial.legendre import leggauss, legval
from triqs.operators import n, c, c_dag
from triqs_cthyb import Solver

beta, n_l = 10.0, 30


def fermions(n_modes):
    """Annihilation operators of n_modes fermions by Jordan-Wigner"""
    a, z, one = np.array([[0.0, 1.0], [0.0, 0.0]]), np.diag([1.0, -1.0]), np.eye(2)
    return [reduce(np.kron, [z] * j + [a] + [one] * (n_modes - j - 1)) for j in range(n_modes)]


def exact_G_l(H, c_imp):
    """G_ij,l = sqrt(2l+1) int_0^beta P_l(2 tau / beta - 1) G_ij(tau), G_ij(tau) = -<c_i(tau) c_j^dagger> by ED"""
    E, W = np.linalg.eigh(H)
    E -= E.min()
    Z = np.exp(-beta * E).sum()
    cs = [W.T @ ci @ W for ci in c_imp]
    x, w = leggauss(200)
    taus = beta * (x + 1) / 2
    G = np.array([[[-np.exp(-(beta - t) * E) @ (ci * cj) @ np.exp(-t * E) / Z for cj in cs] for ci in cs] for t in taus])
    P = np.array([legval(x, np.eye(n_l)[l]) for l in range(n_l)])  # P_l(x) at the nodes
    return np.einsum('l,lt,t,tij->lij', np.sqrt(2 * np.arange(n_l) + 1), P, w * beta / 2, G)


def g_bath(eps):
    return np.array([-np.exp(-eps * t) / (1 + np.exp(-beta * eps)) for t in np.linspace(0, beta, 10001)])


def solve(gf_struct, Delta, h_int, h_loc0, seed, n_cycles=40000, **params):
    S = Solver(beta=beta, gf_struct=gf_struct, n_iw=200, n_tau=10001, n_l=n_l, delta_interface=True)
    for (name, delta), D in zip(S.Delta_tau, Delta):
        delta.data[:] = D
    S.solve(h_int=h_int, h_loc0=h_loc0, n_cycles=n_cycles, n_warmup_cycles=2000, length_cycle=50, random_seed=seed, measure_G_tau=False,
            measure_G_l=True, perform_post_proc=False, verbosity=0, **params)
    return S


# ---- 1. Two blocks of size 1, U n_up n_do: pauli_prob = 0, 0.5, 1 (the default) ----

U, mu, V, eps = 2.0, 0.7, 1.0, 0.3
cf = fermions(4)  # 0, 1: impurity up, do; 2, 3: bath up, do
num = [x.T @ x for x in cf]
H = U * num[0] @ num[1] - mu * (num[0] + num[1]) + eps * (num[2] + num[3])
H += V * sum(cf[s].T @ cf[2 + s] + cf[2 + s].T @ cf[s] for s in range(2))
G_l_ed = exact_G_l(H, [cf[0]])[:, 0, 0]

Delta = [V**2 * g_bath(eps)[:, None, None]] * 2
for pauli_prob in [0.0, 0.5, None]:
    params = {} if pauli_prob is None else {'pauli_prob': pauli_prob}
    S = solve([('up', 1), ('do', 1)], Delta, U * n('up', 0) * n('do', 0), -mu * (n('up', 0) + n('do', 0)), 4271, **params)
    G_l = 0.5 * (S.G_l['up'].data[:, 0, 0] + S.G_l['do'].data[:, 0, 0]).real
    dev = np.abs(G_l - G_l_ed).max()
    print(f"blocks of size 1, pauli_prob = {pauli_prob}: max |G_l - ED| = {dev:.4f} (|G_0| = {abs(G_l_ed[0]):.3f}),"
          f" average order {S.average_order:.2f}")
    assert dev < 0.03, f"pauli_prob = {pauli_prob}: G_l deviates from ED by {dev}"

# ---- 2. One block of size 2 with a hopping t in h_loc: pauli_prob = 0.5 ----
# The off-diagonal hybridization is kept small: with t it causes a sign problem (here the sign is 0.8), and its noise grows
# with it. It cannot be zero, though: with t != 0 and Delta_01 = 0 the solver misses G by a few percent whatever the
# proposal (pauli_prob = 0 does too). Only the diagonal of G is compared, and only for the l < 10 that carry the signal,
# since the noise of G_l is white in l.

U, mu, t = 1.5, 0.5, 0.4
eps_b, Vm = [-0.4, 0.6], np.array([[0.9, 0.05], [0.05, 0.8]])  # bath site k couples to impurity orbital i with Vm[i, k]
cf = fermions(4)  # 0, 1: impurity orbitals; 2, 3: bath sites
num = [x.T @ x for x in cf]
H = U * num[0] @ num[1] - mu * (num[0] + num[1]) + t * (cf[0].T @ cf[1] + cf[1].T @ cf[0])
H += sum(eps_b[k] * num[2 + k] for k in range(2))
H += sum(Vm[i, k] * (cf[i].T @ cf[2 + k] + cf[2 + k].T @ cf[i]) for i in range(2) for k in range(2))
G_l_ed = exact_G_l(H, [cf[0], cf[1]])

Delta = [sum(np.einsum('i,j->ij', Vm[:, k], Vm[:, k])[None, :, :] * g_bath(eps_b[k])[:, None, None] for k in range(2))]
h_loc0 = -mu * (n('ab', 0) + n('ab', 1)) + t * (c_dag('ab', 0) * c('ab', 1) + c_dag('ab', 1) * c('ab', 0))
S = solve([('ab', 2)], Delta, U * n('ab', 0) * n('ab', 1), h_loc0, 9137, pauli_prob=0.5, n_cycles=100000)
dev = max(np.abs(S.G_l['ab'].data.real[:10, a, a] - G_l_ed[:10, a, a]).max() for a in range(2))
print(f"one block of size 2, pauli_prob = 0.5: max |G_l - ED| = {dev:.4f} for l < 10 (|G_0| = {np.abs(G_l_ed[0]).max():.3f}),"
      f" average order {S.average_order:.2f}, sign {S.average_sign:.3f}")
assert dev < 0.06, f"block of size 2: G_l deviates from ED by {dev}"
