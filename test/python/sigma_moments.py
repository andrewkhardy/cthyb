# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# The high-frequency moments of Sigma must be those of G = 1 / (iw - eps - Sigma):
# G_2 = eps + Sigma_0 and G_3 = (eps + Sigma_0)^2 + Sigma_1, with matrix products within a block.
# Checked on a two-orbital atom whose hopping makes Sigma_0 off-diagonal, where an element-wise
# Sigma_0^2 is off by O(1). No Monte Carlo: the density matrix is the atomic one.

import unittest
import numpy as np
from triqs.operators import c, c_dag, n
from triqs.atom_diag import AtomDiag, atomic_density_matrix, trace_rho_op
from triqs_cthyb.tail_fit import sigma_high_frequency_moments, green_high_frequency_moments, _comm, _anticomm

GF_STRUCT = [('bl', 2)]


def moments(eps, h_int, beta=5.0):
    h_0 = sum(eps[i, j] * c_dag('bl', i) * c('bl', j) for i in range(2) for j in range(2))
    h = h_0 + h_int
    ad = AtomDiag(h, [('bl', 0), ('bl', 1)])
    rho = atomic_density_matrix(ad, beta)
    sigma = sigma_high_frequency_moments(rho, ad, GF_STRUCT, h_int)['bl']
    G_2 = green_high_frequency_moments(rho, ad, GF_STRUCT, h)['bl'][2]
    G_3 = np.array([[trace_rho_op(rho, _anticomm(_comm(h, _comm(h, c('bl', a))), c_dag('bl', b)), ad) for b in range(2)]
                    for a in range(2)])
    return sigma, G_2, G_3


class TestSigmaMoments(unittest.TestCase):

    def check(self, eps, h_int):
        sigma, G_2, G_3 = moments(eps, h_int)
        np.testing.assert_allclose(G_2, eps + sigma[0], atol=1e-12)
        np.testing.assert_allclose(G_3, G_2 @ G_2 + sigma[1], atol=1e-12)
        return sigma

    def test_density_density(self):
        sigma = self.check(np.array([[0.0, 0.7], [0.7, 0.3]]), 2.0 * n('bl', 0) * n('bl', 1))
        self.assertGreater(abs(sigma[0][0, 1]), 0.1)  # the case the matrix product matters for

    def test_pair_hopping(self):
        h_int = 2.0 * n('bl', 0) * n('bl', 1) + 0.6 * (c_dag('bl', 0) * c('bl', 1) * c_dag('bl', 0) * c('bl', 1)
                                                      + c_dag('bl', 1) * c('bl', 0) * c_dag('bl', 1) * c('bl', 0))
        self.check(np.array([[0.1, 0.5], [0.5, -0.2]]), h_int)

    def test_diagonal(self):
        self.check(np.array([[0.2, 0.0], [0.0, -0.1]]), 2.0 * n('bl', 0) * n('bl', 1))


if __name__ == '__main__':
    unittest.main()
