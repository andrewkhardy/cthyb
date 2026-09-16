# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# Tests the instantaneous (constant) offset of a retarded interaction: kprime_0, static_shift and
# half_filling_mu in triqs_cthyb.dynamical_interactions. These exist so a script can add the
# offset to mu explicitly, instead of reading it back from what solve() happened to route through
# the analytic path (which is route dependent, and would give a lang_firsov=True run and a
# lang_firsov=False run different chemical potentials).
#
# No solver is constructed and no Monte Carlo runs: this is pure quadrature and linear algebra.

import unittest
import numpy as np
from triqs_cthyb.dynamical_interactions import kprime_0, kprime_0_boson, half_filling_mu, static_shift

BETA, OMEGA_0, N_TAU = 10.0, 1.0, 2001
TAU = np.linspace(0.0, BETA, N_TAU)
Q = -np.cosh(OMEGA_0 * (TAU - BETA / 2)) / (2 * OMEGA_0 * np.sinh(OMEGA_0 * BETA / 2))


class TestDynStaticShift(unittest.TestCase):

    def test_quadrature_matches_closed_form(self):
        """K'(0) from the definition must match coeff/(2 omega_0^2) for the single-boson kernel."""
        for coeff in (1.0, 0.25, -0.7):
            self.assertAlmostEqual(kprime_0(coeff * Q, BETA), kprime_0_boson(coeff, OMEGA_0), places=9)

    def test_independent_of_tau_grid(self):
        """The offset is a two-moment functional, so it must not drift with the tau grid."""
        for n_tau in (401, 2001, 10001):
            tau = np.linspace(0.0, BETA, n_tau)
            q = -np.cosh(OMEGA_0 * (tau - BETA / 2)) / (2 * OMEGA_0 * np.sinh(OMEGA_0 * BETA / 2))
            self.assertAlmostEqual(kprime_0(q, BETA), 1.0 / (2 * OMEGA_0**2), places=8)

    def test_zero_frequency_shortcut_only_for_symmetric_kernels(self):
        """-(1/2) int D is K'(0) only when D is symmetric about beta/2 (benchmark/dynamic_int/
        holstein.py relies on this). It must fail for an asymmetric kernel, or that reliance
        would be silently wrong wherever the kernel is not a plain boson propagator."""
        shortcut = lambda D: -0.5 * np.trapezoid(D, TAU)
        self.assertAlmostEqual(shortcut(Q), kprime_0(Q, BETA), places=5)     # symmetric: agrees
        D_asym = -np.exp(-OMEGA_0 * TAU)
        self.assertGreater(abs(shortcut(D_asym) - kprime_0(D_asym, BETA)), 0.1)  # asymmetric: does not

    def test_reproduces_kanamori_phonon_reference(self):
        """Must reproduce benchmark/dynamic_int/ed_reference/model.py, which is validated against
        exact diagonalization, for the non-uniform coupling g = (0.7, 0.3)."""
        U, J, g_orb = 2.0, 0.3, [0.7, 0.3]
        labels = [(s, o) for s in ('up', 'down') for o in range(2)]
        g = np.array([g_orb[o] for _, o in labels])
        n_so = len(labels)

        kanamori = np.zeros((n_so, n_so))
        for a, (s1, o1) in enumerate(labels):
            for b, (s2, o2) in enumerate(labels):
                if a == b: continue
                kanamori[a, b] = U if o1 == o2 else (U - 3 * J if s1 == s2 else U - 2 * J)

        # The phonon gives D_ab = g_a g_b Q(tau) on every ordered pair, diagonal included
        vertices = [(a, b, g[a] * g[b] * Q) for a in range(n_so) for b in range(n_so)]
        W_shift, level_shift = static_shift(vertices, n_so, BETA)
        mu = half_filling_mu(kanamori, W_shift, level_shift)

        # model.py: W_ab = kanamori_ab - g_a g_b/omega^2, mu = 0.5 W.sum(axis=1) - g_a^2/(2 omega^2)
        W_ref = np.array([[0.0 if a == b else kanamori[a, b] - g[a] * g[b] / OMEGA_0**2
                           for b in range(n_so)] for a in range(n_so)])
        mu_ref = 0.5 * W_ref.sum(axis=1) - g**2 / (2 * OMEGA_0**2)
        np.testing.assert_allclose(kanamori + W_shift, W_ref, atol=1e-8)
        np.testing.assert_allclose(mu, mu_ref, atol=1e-8)
        # mu must differ between the two orbitals -- a sign slip in half_filling_mu makes it
        # come out uniform for exactly these numbers, which looks plausible and is not.
        self.assertGreater(abs(mu[0] - mu[1]), 0.3)

    def test_reproduces_single_orbital_holstein(self):
        """benchmark/dynamic_int/holstein.py's mu = U/2 - g^2/omega_0^2."""
        U, g = 4.0, 0.5
        vertices = [(a, b, g * g * Q) for a in range(2) for b in range(2)]
        W_shift, level_shift = static_shift(vertices, 2, BETA)
        mu = half_filling_mu(np.array([[0.0, U], [U, 0.0]]), W_shift, level_shift)
        np.testing.assert_allclose(mu, U / 2 - g**2 / OMEGA_0**2, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
