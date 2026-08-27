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

    Pair-hopping has no dynamical analogue here: add_dyn_vertex requires each side to
    reduce to a single fermion bilinear (one creation, one annihilation -- see
    dynamical_interactions.hpp), but pair-hopping's c^dag_{a,up} c^dag_{a,down} is a pair
    creation operator, not a bilinear, so it cannot be expressed this way. Representing
    it would need a genuinely different vertex type (op_desc_pair_t, insert_dyn.cpp and
    remove_dyn.cpp all assume one creation + one annihilation per side) -- not attempted
    here, and there is no `pair_hopping` argument.

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
