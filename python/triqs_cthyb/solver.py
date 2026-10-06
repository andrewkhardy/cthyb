################################################################################
#
# TRIQS: a Toolbox for Research in Interacting Quantum Systems
#
# Copyright (C) 2017 by H. UR Strand, P. Seth, I. Krivenko,
#                       M. Ferrero, O. Parcollet
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
the triqs_cthyb solver class
"""
from .solver_core import SolverCore, ConstrParametersT, SolveParametersT
from triqs.gfs import *
import triqs.utility.mpi as mpi
import numpy as np
from itertools import product
from triqs.operators.util.extractors import extract_h_dict, block_matrix_from_op
from .tail_fit import tail_fit as cthyb_tail_fit
from .tail_fit import sigma_high_frequency_moments, green_high_frequency_moments
from .util import orbital_occupations

class Solver(SolverCore):

    def __init__(self, beta, gf_struct, n_iw=1025, n_tau=10001, n_l=30, delta_interface = False, n_tau_bosonic=10001):
        """
        Initialise the solver.

        Parameters
        ----------
        beta : scalar
               Inverse temperature.
        gf_struct : list of pairs [ (str,int), ...]
                    Structure of the Green's functions. It must be a
                    list of pairs, each containing the name of the
                    Green's function block and its linear size.
                    For example: ``[ ('up', 3), ('down', 3) ]``.
        n_iw : integer, optional
               Number of Matsubara frequencies used for the Green's functions.
        n_tau : integer, optional
               Number of imaginary time points used for the Green's functions.
        n_l : integer, optional
             Number of legendre polynomials to use in accumulations of the Green's functions.
        delta_interface: bool, optional
            Are Delta_tau and Delta_infty provided as input instead of G0_iw?
        n_tau_bosonic : integer, optional
            Number of imaginary time points for bosonic Green's functions (D0_tau, Jperp_tau).
        """

        gf_struct = fix_gf_struct_type(gf_struct)

        # Initialise the core solver
        SolverCore.__init__(self, ConstrParametersT(beta=beta, gf_struct=gf_struct,
                            n_iw=n_iw, n_tau=n_tau, n_l=n_l, delta_interface=delta_interface,
                            n_tau_bosonic=n_tau_bosonic))

        mesh = MeshImFreq(beta = beta, statistic="Fermion", n_iw = n_iw)
        self.Sigma_iw = BlockGf(mesh = mesh, gf_struct = gf_struct)
        self.Sigma_iw.zero()
        self.Sigma_iw_raw = None
        self.G_iw = self.Sigma_iw.copy()
        self.G_iw_raw = None
        self.gf_struct = gf_struct
        self.n_iw = n_iw
        self.n_tau = n_tau
        self.delta_interface = delta_interface
        self.G_moments = None
        self.Sigma_moments = None
        self.Sigma_Hartree = None

    def add_dyn_int(self, D_tau, op1, op2):
        r"""
        Add a retarded interaction to the impurity action,

        .. math:: S_\mathrm{dyn} \mathrel{+}= \frac{1}{2} \int_0^\beta d\tau \int_0^\beta d\tau'
                  D(\tau - \tau') \, O_1(\tau) \, O_2(\tau').

        Terms from repeated calls add up. For example, a retarded spin-spin interaction
        :math:`\frac{1}{2} \int\int J(\tau - \tau') \, \mathbf{S}(\tau) \cdot \mathbf{S}(\tau')` is::

            S.add_dyn_int(J_tau, Sz, Sz)
            S.add_dyn_int(J_tau / 2, Sp, Sm)
            S.add_dyn_int(J_tau / 2, Sm, Sp)

        and a phonon coupled to :math:`X = \sum_a g_a n_a` is ``S.add_dyn_int(Q_tau, X, X)``.

        ``solve()`` decides which terms to resum analytically (Lang-Firsov: couplings between
        densities conserved by h_loc) and which to sample, and prints the split. Afterwards the
        two sets are in ``lang_firsov_vertex_operators`` and ``dyn_vertex_operators``.

        Parameters
        ----------
        D_tau : Gf or array
            The real coupling :math:`D(\tau)` on :math:`[0, \beta]`: a scalar_valued or (1, 1) Gf
            on a bosonic imaginary-time mesh, or an array of ``n_tau_bosonic`` values.
        op1, op2 : Operator
            Quadratic operators, i.e. any linear combination of ``c_dag(a) * c(b)``.
        """
        from .dynamical_interactions import _as_scalar_gf, bilinear, expand_dyn_int

        if isinstance(D_tau, Gf):
            D = _as_scalar_gf(D_tau)
        else:
            D = Gf(mesh=self.Jperp_tau.mesh, target_shape=[])
            values = np.asarray(D_tau)
            if values.shape != D.data.shape:
                raise ValueError(f"add_dyn_int: expected {len(D.mesh)} values of D(tau) (n_tau_bosonic), got shape {values.shape}")
            D.data[:] = values
        if np.abs(D.data.imag).max() > 1e-12:
            raise ValueError("add_dyn_int: the coupling D(tau) must be real")
        if np.abs(D.data).max() == 0.0:
            return

        for (key1, key2), coeff in expand_dyn_int(op1, op2).items():
            if abs(np.imag(coeff)) > 1e-12:
                raise ValueError(f"add_dyn_int: complex coefficient {coeff} for "
                                 f"{bilinear(key1)} (x) {bilinear(key2)}; the coupling must stay real")
            self.add_dyn_vertex(bilinear(key1), bilinear(key2), np.real(coeff) * D)

    def solve(self, **params_kw):
        r"""
        Solve the impurity problem for a given G0_iw. If ``measure_G_tau``
        (default = ``True``), ``G_iw`` and ``Sigma_iw`` will be calculated and
        their tails fitted. In addition to the solver parameters, parameters to
        control the tail fitting can be provided.

        Moreover, the fundamental property :math:`G_{ij}(i \omega_n) =
        G_{ji}^*(- i \omega_n)` of the input G0_iw is enforced within C++, and
        a warning is printed if the property was not satisfied. Additionally, if
        ``measure_G_tau`` is set to ``True``, the property :math:`G_{ij}(\tau)=
        G_{ji}^*(\tau)` will be also ensured for the measured :math:`G(\tau)`.
	The difference between the original :math:`G(\tau)` and the hermitized
	:math:`G(\tau)` is stored in the object ``asymmetry_G_tau`` of the solver instance.


        Parameters
        ----------
        params_kw : dict {'param':value} that is passed to the core solver.
                     Two required :class:`parameters <triqs_cthyb.solver_core.SolveParametersT>` are
                        * `h_int` (:ref:`Operator object <triqslibs:operators>`): the local Hamiltonian of the impurity problem to be solved,
                        * `n_cycles` (int): number of measurements to be made.
        perform_post_proc : boolean, optional, default = ``True``
                            Should ``G_iw`` and ``Sigma_iw`` be calculated?
        perform_tail_fit : boolean, optional, default = ``False``
                           Should the tails of ``Sigma_iw`` and ``G_iw`` be fitted?
        fit_max_moment : integer, optional, default = 3
                         Highest moment to fit in the tail of ``Sigma_iw``.
        fit_known_moments : ``ndarray.shape[order, Sigma_iw[0].target_shape]``, optional, default = None
                            Known moments of Sigma_iw, given as an numpy ndarray
        fit_min_n : integer, optional, default = ``int(0.8 * self.n_iw)``
                    Index of ``iw`` from which to start fitting.
        fit_max_n : integer, optional, default = ``n_iw``
                    Index of ``iw`` to fit until.
        """

        # -- Deprecation checks for measure parameters

        depr_params = dict(
            measure_g_tau='measure_G_tau',
            measure_g_l='measure_G_l',
            )

        for key in list(depr_params.keys()):
            if key in list(params_kw.keys()):
                print('WARNING: cthyb.solve parameter %s is deprecated use %s.' % \
                    (key, depr_params[key]))
                val = params_kw.pop(key)
                params_kw[depr_params[key]] = val

        # -- Tail post proc flags

        perform_post_proc = params_kw.pop("perform_post_proc", True)
        perform_tail_fit = params_kw.pop("perform_tail_fit", False)

        if perform_post_proc and perform_tail_fit:
            # If tail parameters provided for Sigma_iw fitting, use them, otherwise use defaults
            if not (("fit_min_n" in params_kw) or ("fit_max_n" in params_kw) or ("fit_max_w" in params_kw) or ("fit_min_w" in params_kw)):
                if mpi.is_master_node():
                    warning = ("!------------------------------------------------------------------------------------!\n"
                               "! WARNING: Using default high-frequency tail fitting parameters in the CTHYB solver. !\n"
                               "! You should check that the fitting range is suitable for your calculation!          !\n"
                               "!------------------------------------------------------------------------------------!")
                    print(warning)
            fit_min_n = params_kw.pop("fit_min_n", None)
            fit_max_n = params_kw.pop("fit_max_n", None)
            fit_min_w = params_kw.pop("fit_min_w", None)
            fit_max_w = params_kw.pop("fit_max_w", None)
            fit_max_moment = params_kw.pop("fit_max_moment", None)
            fit_known_moments = params_kw.pop("fit_known_moments", None)

        # Call the core solver's solve routine
        solve_status = SolverCore.solve(self, SolveParametersT(**params_kw))

        # Post-processing:
        # (only supported for G_tau, to permit compatibility with dft_tools)
        if perform_post_proc and (self.last_solve_parameters.measure_G_tau == True):

            if self.last_solve_parameters.measure_density_matrix:
                # we have the density matrix, so we will compute the high frequency
                # moments and orbital occupations

                self.orbital_occupations = orbital_occupations(self.density_matrix,
                                                               self.gf_struct,
                                                               self.h_loc_diagonalization
                                                               )

                h_int = self.last_solve_parameters.h_int
                self.Sigma_moments = sigma_high_frequency_moments(self.density_matrix,
                                                 self.h_loc_diagonalization,
                                                 self.gf_struct,
                                                 h_int
                                                 )

                self.Sigma_Hartree = {bl: sigma_bl[0] for bl, sigma_bl in self.Sigma_moments.items()}

                self.G_moments = green_high_frequency_moments(self.density_matrix,
                                                         self.h_loc_diagonalization,
                                                         self.gf_struct,
                                                         self.h_loc
                                                         )

            # Fourier transform G_tau to obtain G_iw
            for bl, g in self.G_tau:
                bl_size = g.target_shape[0]
                if self.G_moments is None:
                    known_moments = make_zero_tail(g, 4)
                    known_moments[1] = np.eye(bl_size)
                else:
                    known_moments = self.G_moments[bl]
                self.G_iw[bl].set_from_fourier(g, known_moments)

            assert is_gf_hermitian(self.G_iw)
            self.G_iw_raw = self.G_iw.copy()

            if self.delta_interface:
                G0_iw = self.G_iw.copy()
                Delta_iw = make_gf_from_fourier(self.Delta_tau, self.n_iw)
                h_loc0_mat = block_matrix_from_op(self.h_loc0, self.gf_struct)
                for bl, bl_name in enumerate(self.Delta_tau.indices):
                    G0_iw[bl_name] << inverse(iOmega_n - Delta_iw[bl_name] - h_loc0_mat[bl])
            else:
                G0_iw = self.G0_iw

            # Solve Dyson's eq to obtain Sigma_iw and G_iw and fit the tail
            self.Sigma_iw = dyson(G0_iw=G0_iw, G_iw=self.G_iw)
            self.Sigma_iw_raw = self.Sigma_iw.copy()

            if perform_tail_fit:

                if fit_known_moments is None and self.Sigma_moments is not None:
                    fit_known_moments = self.Sigma_moments

                cthyb_tail_fit(
                    Sigma_iw=self.Sigma_iw,
                    fit_min_n = fit_min_n, fit_max_n = fit_max_n,
                    fit_min_w = fit_min_w, fit_max_w = fit_max_w,
                    fit_max_moment = fit_max_moment,
                    fit_known_moments = fit_known_moments,
                    )

                # Recompute G_iw with the fitted Sigma_iw
                self.G_iw = dyson(G0_iw=G0_iw, Sigma_iw=self.Sigma_iw)
            else:

                # Enforce 1/w behavior of G_iw in the tail fit window
                # and recompute Sigma_iw
                for bl, g in self.G_iw:
                    if self.G_moments is None:
                        tail = make_zero_tail(g, 2)
                        tail[1] = np.eye(g.target_shape[0])
                    else:
                        tail = self.G_moments[bl]

                    g.replace_by_tail_in_fit_window(tail)

                self.Sigma_iw = dyson(G0_iw=G0_iw, G_iw=self.G_iw)

        if getattr(self.last_solve_parameters, "measure_D0_corr", False) and getattr(self.last_solve_parameters, "measure_density_matrix", False):
            from triqs.operators import c, c_dag
            from triqs.atom_diag import trace_rho_op

            # Add back the equal-time values <O_i O_j> of the conserved density combinations
            ops = self.conserved_density_operators
            offset = np.array([[trace_rho_op(self.density_matrix, Oi * Oj, self.h_loc_diagonalization).real for Oj in ops] for Oi in ops])
            self.Q_conserved_l.data[0, :, :] += offset
            self.Q_conserved_tau.data[:] += offset

            # Orbital-resolved Q_tau is only measured when every orbital density commutes with h_loc
            if self.Q_tau is not None and self.Q_l is not None:
                for bl1, bl2 in self.Q_tau.indices:
                    size1 = self.Q_tau[bl1, bl2].target_shape[0]
                    size2 = self.Q_tau[bl1, bl2].target_shape[1]
                    offset = np.zeros((size1, size2), dtype=float)
                    for i1 in range(size1):
                        for i2 in range(size2):
                            if bl1 == bl2 and i1 == i2:
                                op = c_dag(bl1, i1) * c(bl1, i1)
                            else:
                                op = c_dag(bl1, i1) * c(bl1, i1) * c_dag(bl2, i2) * c(bl2, i2)
                            offset[i1, i2] = trace_rho_op(self.density_matrix, op, self.h_loc_diagonalization).real
                    
                    self.Q_l[bl1, bl2].data[0, :, :] += offset
                    self.Q_tau[bl1, bl2].data[:] += offset

        return solve_status
