################################################################################
#
# TRIQS: a Toolbox for Research in Interacting Quantum Systems
#
# Copyright (C) 2025 by The Simons Foundation
#
# TRIQS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# TRIQS. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################
r"""
Retarded generalization of the Hubbard-Kanamori interaction, and the static offset of a retarded coupling.

Every vertex is registered with solver.add_dyn_vertex; solve() decides per vertex whether it is resummed
analytically (Lang-Firsov) or sampled stochastically.
"""
from itertools import product
import numpy as np
from triqs.gfs import Gf
from triqs.operators import c, c_dag, n


def _as_scalar_gf(g):
    """g as a scalar_valued Gf (accepts scalar_valued and (1,1) matrix_valued Gfs)."""
    if len(g.target_shape) == 0:
        return g
    if list(g.target_shape) != [1, 1]:
        raise ValueError(f"Expected a scalar_valued or (1,1) matrix_valued Gf, got target_shape {g.target_shape}")
    scalar_g = Gf(mesh=g.mesh, target_shape=[])
    scalar_g.data[:] = g.data[:, 0, 0]
    return scalar_g


def kanamori_dynamical_vertices(solver, spin_names, orb_names, U=None, Uprime=None, J_hund=None, spin_flip=True):
    r"""
    Register the retarded Hubbard-Kanamori interaction on `solver` with add_dyn_vertex:

        sum_{(a1,s1) != (a2,s2)} D^{s1 s2}_{a1 a2}(tau) n_{a1 s1}(tau) n_{a2 s2}(0)
          - sum_{a1 != a2, s} D^J_{a1 a2}(tau)/2 [c^dag_{a1 s} c_{a1 sbar}](tau) [c^dag_{a2 sbar} c_{a2 s}](0)

    with D^{s1 s2} = U if s1 == s2 else Uprime and D^J = J_hund, as in h_int_kanamori. Each ordered pair is its own
    vertex with the full coupling (no 1/2: op1(tau) op2(0) and op2(tau) op1(0) are different correlators). Pair hopping
    is not included.

    Parameters
    ----------
    solver : Solver or SolverCore
    spin_names : list of str
    orb_names : list
        Orbital labels, as in the solver's gf_struct.
    U, Uprime, J_hund : Gf or dict[(a1, a2)] -> Gf, optional
        Couplings D(tau), scalar_valued or (1,1) matrix_valued: one Gf for every pair, or one per ordered pair (a1, a2)
        (absent pairs are zero). Same-spin intra-orbital terms are always excluded.
    spin_flip : bool
        Include the spin-flip terms (needs J_hund).
    """

    def coupling_for(table, a1, a2):
        if table is None:
            return None
        if isinstance(table, dict):
            return table.get((a1, a2))
        return table

    for s1, s2 in product(spin_names, spin_names):
        table = U if s1 == s2 else Uprime
        for a1, a2 in product(orb_names, orb_names):
            if s1 == s2 and a1 == a2:
                continue
            coupling = coupling_for(table, a1, a2)
            if coupling is None:
                continue
            solver.add_dyn_vertex(n(s1, a1), n(s2, a2), _as_scalar_gf(coupling))

    if spin_flip:
        for s1, s2 in product(spin_names, spin_names):
            if s1 == s2:
                continue
            for a1, a2 in product(orb_names, orb_names):
                if a1 == a2:
                    continue
                coupling = coupling_for(J_hund, a1, a2)
                if coupling is None:
                    continue
                solver.add_dyn_vertex(c_dag(s1, a1) * c(s2, a1), c_dag(s2, a2) * c(s1, a2), _as_scalar_gf(-0.5 * coupling))


# The static offset of a retarded coupling D: K'' = D with K(0) = K(beta) = 0 gives
# K'(0) = -(1/beta) int_0^beta (beta - tau) D(tau) dtau. It depends only on the input coupling, not on how solve()
# routes the vertex, so mu can be set from it. Only for a kernel symmetric about beta/2 does it equal -D(i nu = 0)/2.


def kprime_0(D_tau, beta):
    """K'(0) of a coupling on a uniform tau grid including both endpoints (Gf or array), by Simpson's rule
    (trapezoidal for an even number of points)."""
    D = np.asarray(getattr(D_tau, 'data', D_tau)).real.squeeze()
    n = D.size
    tau = np.linspace(0.0, beta, n)
    integrand = (beta - tau) * D
    if n % 2 == 1:
        w = np.ones(n)
        w[1:-1:2], w[2:-1:2] = 4.0, 2.0
        integral = (beta / (n - 1)) / 3.0 * np.dot(w, integrand)
    else:
        integral = np.trapezoid(integrand, tau)
    return -integral / beta


def kprime_0_boson(coeff, omega_0):
    r"""Exact kprime_0 of coeff * Q(tau), Q(tau) = -cosh(omega_0 (tau - beta/2)) / (2 omega_0 sinh(omega_0 beta/2)):
    coeff / (2 omega_0**2), independent of beta."""
    return coeff / (2.0 * omega_0**2)


def static_shift(vertices, n_orb, beta):
    r"""The static shift solve() applies for the ordered vertices (a, b, D_tau), a and b linear orbital indices (pass both
    orderings): each shifts the level of a by -K'(0) if a == b, and otherwise W_ab and W_ba by -K'(0).

    Returns (W_shift, level_shift): the change of the density-density interaction, counted once per unordered pair,
    and of the orbital energies (a shift of eps, not of mu).
    """
    W_shift = np.zeros((n_orb, n_orb))
    level_shift = np.zeros(n_orb)
    for a, b, D_tau in vertices:
        Kp = kprime_0(D_tau, beta)
        if a == b:
            level_shift[a] -= Kp
        else:
            W_shift[a, b] -= Kp
            W_shift[b, a] -= Kp
    return W_shift, level_shift


def half_filling_mu(W_static, W_shift, level_shift):
    r"""Particle-hole symmetric mu_a = (1/2) sum_{b != a} W_ab + level_shift_a of
    H = sum_{a<b} W_ab n_a n_b + sum_a (level_shift_a - mu_a) n_a, W = W_static + W_shift.

    W_static is the static density-density matrix once per unordered pair, with zero diagonal
    (e.g. dict_to_matrix(extract_U_dict2(h_int), gf_struct)).
    """
    W = np.asarray(W_static) + np.asarray(W_shift)
    return 0.5 * W.sum(axis=1) + np.asarray(level_shift)
