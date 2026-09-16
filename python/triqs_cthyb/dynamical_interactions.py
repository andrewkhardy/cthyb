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
Dynamical (retarded) generalization of the Hubbard-Kanamori interaction.

Mirrors triqs.operators.util.hamiltonians.h_int_kanamori's conventions term-for-term,
but every coupling constant is a retarded propagator D(tau) instead of a static number,
and every term is registered via solver.add_dyn_vertex(...) instead of being added to
h_int. Which mechanism actually solves each resulting vertex -- the analytic Lang-Firsov
resummation, or the stochastic double expansion -- is decided automatically, per vertex,
when solver.solve() runs; see dynamical_interactions.hpp for how.
"""
from itertools import product
import numpy as np
from triqs.gfs import Gf, MeshImTime
from triqs.operators import c, c_dag, n


def _as_scalar_gf(g):
    """add_dyn_vertex needs a scalar_valued Gf; accept the more common (1,1) matrix_valued
    shape too (e.g. GfImTime(indices=[0], ...)) and convert."""
    if len(g.target_shape) == 0:
        return g
    if list(g.target_shape) != [1, 1]:
        raise ValueError(f"Expected a scalar_valued or (1,1) matrix_valued Gf, got target_shape {g.target_shape}")
    scalar_g = Gf(mesh=g.mesh, target_shape=[])
    scalar_g.data[:] = g.data[:, 0, 0]
    return scalar_g


def kanamori_dynamical_vertices(solver, spin_names, orb_names, U=None, Uprime=None, J_hund=None, spin_flip=True):
    r"""
    Register the dynamical generalization of a Hubbard-Kanamori interaction on `solver`
    via repeated calls to solver.add_dyn_vertex(...):

        H(tau) = sum_{(a1,s1) != (a2,s2)} D_{a1 a2}^{s1 s2}(tau) n_{a1 s1}(tau) n_{a2 s2}(0)
               - sum_{a1 != a2} D_{a1 a2}^J(tau) [c^dag_{a1,s}c_{a1,sbar}](tau) [c^dag_{a2,sbar}c_{a2,s}](0)

    where D^{s1 s2}_{a1 a2} = U_{a1 a2} if s1 == s2, else Uprime_{a1 a2} -- the same
    convention as h_int_kanamori -- and D^J is J_hund. This is the static Hamiltonian's
    density-density and spin-flip terms with each coupling constant promoted to a
    retarded propagator; call h_int_kanamori for the static part of h_int as usual.

    Note the density-density term carries no 1/2 prefactor, unlike h_int_kanamori's
    own H = (1/2) sum_{i!=j} U_ij n_i n_j: there, n_i(0)*n_j(0) and n_j(0)*n_i(0) are
    the *same* instantaneous operator, so visiting each unordered pair from both
    orderings with half weight each is just bookkeeping. Here, op1(tau)*op2(0) is a
    retarded object -- the (a1,a2) and (a2,a1) orderings are *different* correlators,
    not the same term counted twice -- so each ordered pair needs the coupling at full
    weight. (The spin-flip term below is different: it matches
    expand_Jperp_into_vertices's validated coupling/2-with-both-orderings convention,
    so it keeps its 1/2.)

    Pair-hopping is not registered here, but not because it cannot be: reordering
    c^dag_{a1,up} c^dag_{a1,down} c_{a2,down} c_{a2,up} gives
    (c^dag_{a1,up} c_{a2,up}) (c^dag_{a1,down} c_{a2,down}), a product of two ordinary
    bilinears, which is exactly what add_dyn_vertex takes. A retarded pair-hopping term
    can therefore be registered directly with add_dyn_vertex; it is not a density
    coupling, so it would be sampled stochastically. This helper simply does not add it
    (there is no `pair_hopping` argument), and no test covers it yet.

    Parameters
    ----------
    solver : triqs_cthyb.Solver or SolverCore
        The solver to register vertices on; call before solver.solve().
    spin_names : list of str
        e.g. ['up', 'down'].
    orb_names : list
        Orbital labels, matching the second index used in solver's gf_struct.
    U, Uprime : Gf, or dict[(a1, a2)] -> Gf, optional
        Retarded density-density couplings D_{a1,a2}(tau): U for same-spin pairs,
        Uprime for opposite-spin pairs (intra-orbital same-spin, a1==a2, is always
        excluded -- Pauli exclusion makes n_{a,s}^2 = n_{a,s}, not a genuine
        interaction, exactly as in h_int_kanamori). A single Gf is broadcast to every
        a1 != a2 pair; a dict gives one coupling per unordered pair, with pairs absent
        from the dict treated as zero. Either matrix_valued (1,1) or scalar_valued Gfs
        are accepted.
    J_hund : Gf, or dict[(a1, a2)] -> Gf, optional
        Retarded spin-flip coupling D^J_{a1,a2}(tau), keyed the same way as U/Uprime.
        Required if spin_flip is True.
    spin_flip : bool
        Include the dynamical spin-flip terms (default True; needs J_hund).
    """

    def coupling_for(table, a1, a2):
        if table is None:
            return None
        if isinstance(table, dict):
            return table.get((a1, a2))
        return table  # single Gf, broadcast to every pair

    # Density-density: same-spin uses U, opposite-spin uses Uprime. Each ordered pair
    # (a1,s1) != (a2,s2) is its own retarded vertex and gets the coupling at full
    # weight (see the no-1/2-prefactor note in the docstring above); both orderings of
    # each unordered pair are registered separately, as different correlators.
    for s1, s2 in product(spin_names, spin_names):
        table = U if s1 == s2 else Uprime
        for a1, a2 in product(orb_names, orb_names):
            if s1 == s2 and a1 == a2:
                continue
            coupling = coupling_for(table, a1, a2)
            if coupling is None:
                continue
            solver.add_dyn_vertex(n(s1, a1), n(s2, a2), _as_scalar_gf(coupling))

    # Spin-flip: c^dag_{s1,a1} c_{s2,a1} (tau) * c^dag_{s2,a2} c_{s1,a2} (0), a1 != a2,
    # s1 != s2 -- matches h_int_kanamori's spin-flip term (same -0.5*J_hund factor).
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


# =======================================================================================
# The constant (instantaneous) offset of a retarded interaction
# =======================================================================================
# A retarded coupling D(tau) carries a static part. When a vertex is resummed analytically,
# the K-functional regenerates the *full* retarded interaction including that static part, so
# solve() subtracts it from h_loc to avoid double counting; a vertex sampled stochastically is
# not touched, because insert_dyn samples D(tau) as written. That subtraction therefore depends
# on how each vertex was routed, and has to stay inside solve().
#
# What does NOT belong inside solve() is choosing mu. The offset relevant to the filling is a
# property of the *input* vertex list alone, identical whether a vertex ends up analytic or
# stochastic, and these functions compute it exactly so it can be added to mu explicitly.
#
#   K'' = D with K(0) = K(beta) = 0  =>  K'(0) = -(1/beta) int_0^beta (beta - tau) D(tau) dtau.
#
# Because (beta - tau) is degree 1 in x = 2 tau/beta - 1, only the l = 0 and l = 1 Legendre
# moments contribute -- P_{l>=2} are orthogonal to every degree-1 polynomial. So the two-term
# form K'(0) = -(beta/2)(d_0 - d_1/3) used internally is an identity, not a truncation, and
# `dyn_n_l` cannot change the offset.
#
# WARNING. For a kernel symmetric about beta/2 -- every ordinary boson propagator -- the above
# collapses to K'(0) = -(1/2) int_0^beta D dtau = -(1/2) D(i nu = 0), which is why a
# zero-frequency shortcut works. It is NOT valid for an asymmetric kernel (a charged/complex
# boson, or a D(tau) from a self-consistent loop): there the shortcut can be off by tens of
# percent. Use kprime_0, not the shortcut.


def kprime_0(D_tau, beta):
    r"""K'(0) of a tau-sampled retarded coupling: the instantaneous part that has to be
    accounted for in mu and in the static interaction.

    Parameters
    ----------
    D_tau : Gf or 1d array
        The coupling on a uniform tau grid including both endpoints (a scalar_valued or (1,1)
        Gf, or a plain array).
    beta : float

    Notes
    -----
    Simpson quadrature of the definition. The boson kernel is most strongly curved exactly at
    tau = 0 and tau = beta, where the trapezoidal rule is worst, so Simpson is worth the two
    extra lines: at n_tau_bosonic = 2001 the error drops from ~1e-6 to ~1e-10.
    """
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
    r"""Closed form of kprime_0 for a coupling ``coeff * Q(tau)`` with the single-boson kernel

        Q(tau) = -cosh(omega_0 (tau - beta/2)) / (2 omega_0 sinh(omega_0 beta / 2)),

    namely ``coeff / (2 omega_0**2)``. Exact and beta-independent: int_0^beta Q = -1/omega_0**2
    and int_0^beta Q(tau) x(tau) dtau = 0 by symmetry about beta/2. Use it to check kprime_0.
    """
    return coeff / (2.0 * omega_0**2)


def static_shift(vertices, n_orb, beta):
    r"""Fold a vertex list into the static shift it implies, in the convention solve() applies.

    Per *ordered* vertex with Kp = K'(0) of its coupling: an op1, op2 pair on the same orbital a
    shifts that orbital's energy by -Kp; on different orbitals a, b it shifts the density-density
    coupling by -Kp in both (a, b) and (b, a). Registering both orderings -- the convention used
    throughout -- therefore gives W_ab -= 2 Kp, counted once per unordered pair.

    Parameters
    ----------
    vertices : iterable of (a, b, D_tau)
        `a`, `b` are the two orbitals' linear indices (the order of
        ``itertools.product(block_names, range(block_size))``), `D_tau` that ordered vertex's
        coupling. Pass both orderings.
    n_orb : int
    beta : float

    Returns
    -------
    W_shift : (n_orb, n_orb) ndarray
        Added to the static density-density interaction, counted once per unordered pair.
    level_shift : (n_orb,) ndarray
        Added to the orbital energies (i.e. this is -K'(0), a shift of eps, not of mu).
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
    r"""Particle-hole symmetric mu per orbital for the total effective density-density model.

    For H = sum_{a<b} W_ab n_a n_b + sum_a (level_shift_a - mu_a) n_a, particle-hole symmetry
    requires the coefficient of n_a to be -(1/2) sum_{b != a} W_ab, hence

        mu_a = (1/2) sum_{b != a} W_ab + level_shift_a.

    Note `level_shift` enters with a PLUS: it is the shift applied to the orbital energy, not to
    mu. Getting that sign backwards is easy and not always obvious from the result.

    `W_static` is the static h_int's density-density matrix counted once per unordered pair,
    i.e. what ``dict_to_matrix(extract_U_dict2(h_int), gf_struct)`` returns, with zero diagonal.
    Only meaningful when the full model really is particle-hole symmetric -- check the measured
    filling.
    """
    W = np.asarray(W_static) + np.asarray(W_shift)
    return 0.5 * W.sum(axis=1) + np.asarray(level_shift)
