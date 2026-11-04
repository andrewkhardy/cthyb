# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later

# kanamori_dynamical_vertices is the retarded h_int_kanamori term for term: with constant couplings,
# D(tau) -> D delta(tau), each vertex (D, O1, O2) of S = (1/2) int int D O1 O2 contributes (1/2) D O1 O2 to H,
# and the sum must be h_int_kanamori with the same spin_flip and pair_hopping flags. No Monte Carlo.
#
# Before 2026-11-04 the spin flip was half of h_int_kanamori's (coupling -J/2 per vertex) and there was no
# pair hopping. The pair hopping 1/2 J c^dag_{a s} c^dag_{a sbar} c_{b sbar} c_{b s} is registered as the
# particle-hole product [c^dag_{a s} c_{b s}] (x) [c^dag_{a sbar} c_{b sbar}], equal to it with no contraction.

from itertools import product
import numpy as np
from triqs.operators import Operator
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs_cthyb.dynamical_interactions import kanamori_dynamical_vertices


class StaticLimit:
    """Stands in for the Solver: collects (1/2) D O1 O2 for constant D."""

    def __init__(self):
        self.H = Operator()

    def add_dyn_int(self, D, op1, op2):
        self.H += 0.5 * D * op1 * op2


spin_names, orbs, J = ['up', 'down'], [0, 1, 2], 0.3
U = np.array([[0.0, 1.1, 1.3], [1.1, 0.0, 0.9], [1.3, 0.9, 0.0]])        # same spin, a1 != a2
Uprime = np.array([[2.0, 1.5, 1.4], [1.5, 2.2, 1.6], [1.4, 1.6, 1.9]])    # opposite spin
per_pair = lambda M: {(a1, a2): M[a1, a2] for a1, a2 in product(orbs, orbs)}

for spin_flip, pair_hopping in product([False, True], [False, True]):
    S = StaticLimit()
    kanamori_dynamical_vertices(S, spin_names, orbs, U=per_pair(U), Uprime=per_pair(Uprime), J_hund=J,
                                spin_flip=spin_flip, pair_hopping=pair_hopping)
    reference = h_int_kanamori(spin_names, len(orbs), U, Uprime, J, spin_flip=spin_flip, pair_hopping=pair_hopping,
                               off_diag=True)
    difference = S.H - reference
    assert difference.is_zero(), f"spin_flip={spin_flip}, pair_hopping={pair_hopping}: {difference}"
