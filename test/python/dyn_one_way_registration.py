# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# A retarded density-density coupling registered one way, add_dyn_int(D, n_up, n_dn), is the same
# action as D/2 registered both ways, since D(tau) = D(beta - tau). Both go through Lang-Firsov.
#
# The Lang-Firsov weight of a pair of occupation kinks must not depend on which of the two operators
# a move inserts or removes, so the kernel has to be symmetric, K_ab = K_ba. With it, the two
# registrations give the same kernel table and the same static shift of h_loc, and with the same
# seed the same Markov chain: the results agree to round-off. Without it the one-way registration
# weighs a pair by K_ab or by K_ba = 0 depending on the move, which breaks detailed balance.

import numpy as np
from triqs.gfs import Gf, inverse, iOmega_n, Fourier, MeshImFreq
from triqs.operators import n
from triqs_cthyb import Solver

beta, U, mu, V, eps = 5.0, 2.0, 1.0, 1.0, 0.5
omega_0, g = 1.0, 0.8
n_iw, n_tau = 200, 1001


def solve(registrations):
    S = Solver(beta=beta, gf_struct=[('up', 1), ('dn', 1)], n_iw=n_iw, n_tau=n_tau, n_tau_bosonic=n_tau,
               delta_interface=True)
    delta = Gf(mesh=MeshImFreq(beta, 'Fermion', n_iw), target_shape=[1, 1])
    delta << V**2 * inverse(iOmega_n - eps)
    for _, d in S.Delta_tau:
        d << Fourier(delta)

    tau = np.linspace(0.0, beta, n_tau)
    Q = -g**2 * np.cosh(omega_0 * (tau - beta / 2)) / (2 * omega_0 * np.sinh(omega_0 * beta / 2))
    for weight, op1, op2 in registrations:
        S.add_dyn_int(weight * Q, op1, op2)

    S.solve(h_int=U * n('up', 0) * n('dn', 0), h_loc0=-mu * (n('up', 0) + n('dn', 0)),
            n_cycles=3000, n_warmup_cycles=500, length_cycle=50, random_seed=8471,
            measure_density_matrix=True, use_norm_as_weight=True, measure_G_tau=True, verbosity=0)
    return S


one_way = solve([(1.0, n('up', 0), n('dn', 0))])
split = solve([(0.5, n('up', 0), n('dn', 0)), (0.5, n('dn', 0), n('up', 0))])

for (bl, g1), (_, g2) in zip(one_way.G_tau, split.G_tau):
    np.testing.assert_allclose(g1.data, g2.data, rtol=0, atol=1e-10, err_msg=f"G_tau[{bl}]")
for rho1, rho2 in zip(one_way.density_matrix, split.density_matrix):
    np.testing.assert_allclose(rho1, rho2, rtol=0, atol=1e-10)
