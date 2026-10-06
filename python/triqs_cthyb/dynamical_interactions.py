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
Helpers for retarded (dynamical) interactions.

A retarded interaction enters the impurity action as

    S_dyn = 1/2 \int_0^beta dtau \int_0^beta dtau' D(tau - tau') O1(tau) O2(tau'),

with O1, O2 quadratic operators. Register one with ``Solver.add_dyn_int(D_tau, O1, O2)``;
``expand_dyn_int`` is how such a term is broken into the single-bilinear vertices the
C++ solver samples.
"""
from itertools import product
import numpy as np
from triqs.gfs import Gf
from triqs.operators import c, c_dag, n


def _as_scalar_gf(g):
    """Return `g` as a scalar_valued Gf; a (1, 1) matrix_valued Gf is converted."""
    if len(g.target_shape) == 0:
        return g
    if list(g.target_shape) != [1, 1]:
        raise ValueError(f"Expected a scalar_valued or (1,1) matrix_valued Gf, got target_shape {g.target_shape}")
    scalar_g = Gf(mesh=g.mesh, target_shape=[])
    scalar_g.data[:] = g.data[:, 0, 0]
    return scalar_g


def bilinear(key):
    """The operator c^dag_a c_b for a key ((True, a), (False, b)) from expand_dyn_int."""
    (_, a), (_, b) = key
    return c_dag(*a) * c(*b)


def _bilinear_terms(op, name):
    for monomial, coeff in op:
        if len(monomial) != 2 or monomial[0][0] == monomial[1][0]:
            raise ValueError(f"{name} must be quadratic, a sum of c_dag(a)*c(b) terms, but contains "
                             f"the term {coeff} * {monomial}")
        yield tuple((bool(dagger), tuple(indices)) for dagger, indices in monomial), coeff


def expand_dyn_int(op1, op2):
    r"""Expand op1 (x) op2 into products of single bilinears.

    Returns ``{(key1, key2): coeff}`` such that

        op1(tau) op2(tau') = sum coeff * bilinear(key1)(tau) * bilinear(key2)(tau').

    The two operators act at different times, so each is expanded on its own; they are never
    multiplied together as operators.
    """
    terms = {}
    for key1, a in _bilinear_terms(op1, 'op1'):
        for key2, b in _bilinear_terms(op2, 'op2'):
            terms[key1, key2] = terms.get((key1, key2), 0.0) + a * b
    return {k: v for k, v in terms.items() if abs(v) > 1e-14}


def kanamori_dynamical_vertices(solver, spin_names, orb_names, U=None, Uprime=None, J_hund=None, spin_flip=True):
    r"""Register the retarded version of h_int_kanamori's density-density and spin-flip terms.

    For each ordered pair of spin-orbitals (a1, s1) != (a2, s2)

        solver.add_dyn_int(D, n(s1, a1), n(s2, a2))     D = U if s1 == s2 else Uprime

    and, if `spin_flip`, for each a1 != a2 and s1 != s2

        solver.add_dyn_int(-J_hund / 2, c_dag(s1, a1) c(s2, a1), c_dag(s2, a2) c(s1, a2)).

    Both orderings of a pair are registered, each at full weight, so for a static
    D = U delta(tau) this reproduces h_int_kanamori's (1/2) sum_{i != j} U_ij n_i n_j.
    Pair hopping is not included. Call h_int_kanamori for the static part as usual.

    Parameters
    ----------
    solver : triqs_cthyb.Solver
    spin_names : list of str
    orb_names : list
        Orbital labels, the second index of the operators.
    U, Uprime, J_hund : Gf, or dict[(a1, a2)] -> Gf, optional
        Same-spin, opposite-spin and spin-flip couplings D(tau). A single Gf applies to every
        orbital pair; with a dict, missing pairs are zero.
    spin_flip : bool
        Include the spin-flip terms (needs J_hund).
    """

    def coupling_for(table, a1, a2):
        if isinstance(table, dict):
            return table.get((a1, a2))
        return table

    for s1, s2 in product(spin_names, spin_names):
        table = U if s1 == s2 else Uprime
        for a1, a2 in product(orb_names, orb_names):
            if s1 == s2 and a1 == a2:
                continue
            D = coupling_for(table, a1, a2)
            if D is not None:
                solver.add_dyn_int(D, n(s1, a1), n(s2, a2))

    if spin_flip:
        for s1, s2 in product(spin_names, spin_names):
            if s1 == s2:
                continue
            for a1, a2 in product(orb_names, orb_names):
                D = coupling_for(J_hund, a1, a2) if a1 != a2 else None
                if D is not None:
                    solver.add_dyn_int(-0.5 * D, c_dag(s1, a1) * c(s2, a1), c_dag(s2, a2) * c(s1, a2))


# =======================================================================================
# The instantaneous part of a retarded interaction
# =======================================================================================
# A coupling D(tau) has a static part, K'(0), with K'' = D and K(0) = K(beta) = 0:
#
#     K'(0) = -(1/beta) int_0^beta (beta - tau) D(tau) dtau.
#
# When a vertex is resummed analytically (Lang-Firsov), solve() subtracts this from h_loc;
# a vertex sampled stochastically is left alone. That subtraction depends on the routing, so
# mu must not be read back from the solver: compute the offset from the input couplings with
# the functions below and put it into mu yourself.
#
# For a kernel symmetric about beta/2, K'(0) = -(1/2) D(i nu = 0). This shortcut is wrong for
# an asymmetric kernel; kprime_0 is not.


def kprime_0(D_tau, beta):
    """K'(0) of a coupling sampled on a uniform tau grid including both endpoints.

    `D_tau` is a scalar_valued or (1, 1) Gf, or a plain array. Simpson's rule when the number
    of points is odd, the trapezoidal rule otherwise.
    """
    D = np.asarray(getattr(D_tau, 'data', D_tau)).real.squeeze()
    n_pts = D.size
    tau = np.linspace(0.0, beta, n_pts)
    integrand = (beta - tau) * D
    if n_pts % 2 == 1:
        w = np.ones(n_pts)
        w[1:-1:2], w[2:-1:2] = 4.0, 2.0
        integral = (beta / (n_pts - 1)) / 3.0 * np.dot(w, integrand)
    else:
        integral = np.trapezoid(integrand, tau)
    return -integral / beta


def kprime_0_boson(coeff, omega_0):
    """K'(0) of coeff * Q(tau), Q(tau) = -cosh(omega_0 (tau - beta/2)) / (2 omega_0 sinh(omega_0 beta/2)).

    Exact and independent of beta: coeff / (2 omega_0^2).
    """
    return coeff / (2.0 * omega_0**2)


def static_shift(vertices, n_orb, beta):
    r"""The static density-density shift implied by a list of density vertices.

    Parameters
    ----------
    vertices : iterable of (a, b, D_tau)
        One entry per *ordered* vertex n_a(tau) n_b(0); `a`, `b` are linear orbital indices.
    n_orb : int
    beta : float

    Returns
    -------
    W_shift : (n_orb, n_orb) ndarray
        Shift of the static density-density matrix, counted once per unordered pair. Each
        ordered vertex adds -K'(0) to (a, b) and (b, a).
    level_shift : (n_orb,) ndarray
        Shift of the orbital energies (not of mu): -K'(0) for each a == b vertex.
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
    r"""Particle-hole symmetric mu per orbital: mu_a = (1/2) sum_{b != a} W_ab + level_shift_a.

    W = W_static + W_shift, with W_static the static density-density matrix counted once per
    unordered pair and zero on the diagonal (``dict_to_matrix(extract_U_dict2(h_int), gf_struct)``).
    Only meaningful if the full model is particle-hole symmetric; check the measured filling.
    """
    W = np.asarray(W_static) + np.asarray(W_shift)
    return 0.5 * W.sum(axis=1) + np.asarray(level_shift)
