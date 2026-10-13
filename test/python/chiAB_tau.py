"""
The density-density correlator <n_up(tau) n_do(0)>, by measure_chiAB_tau and measure_nn_tau, and with lists of
operators in chi_A_vec and chi_B_vec, against exact diagonalization of the impurity and its discrete bath.

Author: Hugo U.R. Strand (2018) hugo.strand@gmail.com
"""

from functools import reduce
import numpy as np

from triqs.gfs import inverse, iOmega_n
from triqs.operators import n
from triqs_cthyb import Solver

beta, mu, U = 2.1, 2.0, 5.0
V, eps = [2.0, 5.0], [0.0, 4.0]  # one bath site per (V, eps) and spin


def exact_diagonalization():
    """The energies, and n_up and n_do in the eigenbasis, of the impurity and its bath sites."""
    n_modes = 2 * (1 + len(V))  # 0, 1: impurity up, down; 2 + 2k + s: bath site k, spin s
    a, z, one = np.array([[0.0, 1.0], [0.0, 0.0]]), np.diag([1.0, -1.0]), np.eye(2)
    c = [reduce(np.kron, [z] * j + [a] + [one] * (n_modes - j - 1)) for j in range(n_modes)]  # Jordan-Wigner
    num = [cj.T @ cj for cj in c]
    H = U * num[0] @ num[1]
    for s in range(2):
        H -= mu * num[s]
        for k, (Vk, ek) in enumerate(zip(V, eps)):
            b = 2 + 2 * k + s
            H += ek * num[b] + Vk * (c[s].T @ c[b] + c[b].T @ c[s])
    E, W = np.linalg.eigh(H)
    return E - E.min(), W.T @ num[0] @ W, W.T @ num[1] @ W


E, N_UP, N_DO = exact_diagonalization()


def exact(A, B, taus):
    """<A(tau) B(0)> for A and B in the eigenbasis."""
    Z = np.exp(-beta * E).sum()
    return np.array([np.exp(-(beta - t) * E) @ (A * B.T) @ np.exp(-t * E) for t in taus]) / Z


def exact_nn(taus):
    return exact(N_UP, N_DO, taus)


S = Solver(beta=beta, gf_struct=[['up', 1], ['do', 1]], n_iw=30, n_tau=2 * 30 + 1)
for _, g0 in S.G0_iw:
    g0 << inverse(iOmega_n + mu - sum(Vk**2 * inverse(iOmega_n - ek) for Vk, ek in zip(V, eps)))

solve_params = dict(h_int=U * n('up', 0) * n('do', 0), measure_G_tau=True, move_double=True,
                    length_cycle=20, n_warmup_cycles=int(1e4), n_cycles=int(1e5))
S.solve(**solve_params, measure_chiAB_tau=True, chi_A_vec=n('do', 0), chi_B_vec=n('up', 0), measure_nn_tau=True)

# The QMC noise is a few 1e-3 here (at most 0.012 over many seeds); the correlator is ~0.3
taus = np.array([float(t) for t in S.chiAB_tau.mesh])
np.testing.assert_allclose(S.chiAB_tau.data.real, exact_nn(taus), atol=0.03)

taus = np.array([float(t) for t in S.nn_tau['up', 'do'].mesh])
np.testing.assert_allclose(S.nn_tau['up', 'do'].data[:, 0, 0].real, exact_nn(taus), atol=0.03)

# The DLR coefficients give the correlator at any tau, not only on the mesh
taus = np.random.default_rng(1).uniform(0, beta, 50)
np.testing.assert_allclose([S.chiAB_dlr(t).real for t in taus], exact_nn(taus), atol=0.03)
np.testing.assert_allclose([S.nn_dlr['up', 'do'](t)[0, 0].real for t in taus], exact_nn(taus), atol=0.03)

# Lists: chiAB_tau[i, j] = <A_i(tau) B_j(0)>, here with A = [n_up, n_up n_do] and B = [n_up]
S.solve(**solve_params, measure_chiAB_tau=True, chi_A_vec=[n('up', 0), n('up', 0) * n('do', 0)], chi_B_vec=[n('up', 0)])
assert S.chiAB_tau.target_shape == (2, 1) and S.chiAB_dlr.target_shape == (2, 1)
taus = np.array([float(t) for t in S.chiAB_tau.mesh])
for i, A in enumerate([N_UP, N_UP @ N_DO]):
    np.testing.assert_allclose(S.chiAB_tau.data[:, i, 0].real, exact(A, N_UP, taus), atol=0.03)
