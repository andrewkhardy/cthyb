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
from triqs.operators import Operator
from triqs.operators.util.extractors import block_matrix_from_op, op_from_block_matrix
from .tail_fit import tail_fit as cthyb_tail_fit
from .tail_fit import sigma_high_frequency_moments, green_high_frequency_moments
from .util import orbital_occupations

class Solver(SolverCore):

    def __init__(self, beta, gf_struct, n_iw=1025, n_tau=10001, n_l=30, delta_interface = False, n_tau_bosonic=10001,
                 dlr_wmax=None, dlr_eps=1e-10):
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
            In DLR mode: are Delta_iw and h_loc0 provided instead of G0_iw?
        n_tau_bosonic : integer, optional
            Number of imaginary time points for bosonic Green's functions (D0_tau, Jperp_tau).
        dlr_wmax : float, optional
            Setting it switches on the DLR mode: the input G0_iw (or Delta_iw with
            delta_interface) and the output G_iw and Sigma_iw are on the symmetrized
            DLR Matsubara mesh (beta, dlr_wmax, dlr_eps), ``dlr_mesh``. The Monte Carlo
            still works on n_tau points: the input is sampled there, and the measured
            G_tau is fitted back onto the DLR basis (``G_dlr``, ``G_tau_dlr``).
            G, not only Delta, must have no spectral weight above dlr_wmax.
        dlr_eps : float, optional
            Accuracy of the DLR mesh.
        """

        gf_struct = fix_gf_struct_type(gf_struct)
        self.dlr = dlr_wmax is not None

        # Initialise the core solver. In DLR mode the core always uses the Delta interface:
        # solve() sets its Delta_tau (and h_loc0 when the input is G0_iw) from the DLR input.
        SolverCore.__init__(self, ConstrParametersT(beta=beta, gf_struct=gf_struct,
                            n_iw=n_iw, n_tau=n_tau, n_l=n_l, delta_interface=delta_interface or self.dlr,
                            n_tau_bosonic=n_tau_bosonic))

        if self.dlr:
            mesh = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=dlr_wmax, eps=dlr_eps, symmetrize=True)
            self.dlr_mesh = mesh
            # The input with delta_interface, otherwise computed from G0_iw in solve()
            self.Delta_iw = BlockGf(mesh = mesh, gf_struct = gf_struct)
            self._G0_iw_dlr = None if delta_interface else self.Delta_iw.copy()
        else:
            mesh = MeshImFreq(beta = beta, statistic="Fermion", n_iw = n_iw)
        self.Sigma_iw = BlockGf(mesh = mesh, gf_struct = gf_struct)
        self.Sigma_iw.zero()
        self.Sigma_iw_raw = None
        self.G_iw = self.Sigma_iw.copy()
        self.G_iw_raw = None
        self.G_dlr = None
        self.G_tau_dlr = None
        self.crm_failure = None  # why the CRM of the last solve failed, if it did (dlr_sigma='crm')
        self._retarded = False  # set by add_dyn_int
        self._chiAB_scalar = False  # chi_A_vec and chi_B_vec of the last solve were single operators
        self.gf_struct = gf_struct
        self.n_iw = n_iw
        self.n_tau = n_tau
        self.delta_interface = delta_interface
        self.G_moments = None
        self.Sigma_moments = None
        self.Sigma_Hartree = None

    @property
    def G0_iw(self):
        """The non-interacting Green's function, on the DLR mesh in DLR mode."""
        if self.dlr and not self.delta_interface:
            return self._G0_iw_dlr
        return SolverCore.G0_iw.__get__(self)

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
            on a bosonic imaginary-time or DLR (MeshDLR, MeshDLRImTime, MeshDLRImFreq) mesh, or an
            array of ``n_tau_bosonic`` values. A DLR coupling is sampled on ``n_tau_bosonic`` points.
        op1, op2 : Operator
            Quadratic operators, i.e. any linear combination of ``c_dag(a) * c(b)``.
        """
        from .dynamical_interactions import _as_scalar_gf, bilinear, expand_dyn_int

        if isinstance(D_tau, Gf) and isinstance(D_tau.mesh, (MeshDLR, MeshDLRImTime, MeshDLRImFreq)):
            D_tau = make_gf_imtime(D_tau, len(self.Jperp_tau.mesh))
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
        self._retarded = True

        for (key1, key2), coeff in expand_dyn_int(op1, op2).items():
            if abs(np.imag(coeff)) > 1e-12:
                raise ValueError(f"add_dyn_int: complex coefficient {coeff} for "
                                 f"{bilinear(key1)} (x) {bilinear(key2)}; the coupling must stay real")
            self.add_dyn_vertex(bilinear(key1), bilinear(key2), np.real(coeff) * D)

    @property
    def chiAB_tau(self):
        r""":math:`\chi_{AB}(\tau) = \langle A_i(\tau) B_j(0) \rangle` as ``chiAB_tau[i, j]`` on the n_tau mesh, for
        ``measure_chiAB_tau=True`` with the lists ``chi_A_vec`` and ``chi_B_vec``, as in CTINT; a scalar Gf
        :math:`\langle A(\tau) B(0) \rangle` for two single operators."""
        chi = SolverCore.chiAB_tau.__get__(self)
        return chi[0, 0] if chi is not None and self._chiAB_scalar else chi

    @property
    def chiAB_dlr(self):
        """chiAB_tau as DLR coefficients, from which it is evaluated at any tau or Matsubara frequency."""
        chi = SolverCore.chiAB_dlr.__get__(self)
        return chi[0, 0] if chi is not None and self._chiAB_scalar else chi

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
        dlr_sigma : string, optional, default = ``'dyson'``
                    DLR mode only: how ``Sigma_iw`` is obtained on the DLR mesh from the DLR fit of
                    ``G_tau``. ``'dyson'``: :math:`G_0^{-1} - G^{-1}` at each DLR frequency.
                    ``'crm'``: constrained residual minimization of the Dyson equation
                    (``triqs.gfs.dlr_crm_dyson_solver.minimize_dyson``) with the high-frequency
                    moments of Sigma, which needs ``measure_density_matrix``. Where the minimization fails
                    (LinAlgError, ValueError, or a non-finite Sigma), ``Sigma_iw`` is the ``'dyson'`` one, with a
                    warning, and ``crm_failure`` holds the error (None otherwise).
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

        if "measure_O_tau" in params_kw:
            raise ValueError("measure_O_tau=(O_1, O_2) is now measure_chiAB_tau=True, chi_A_vec=O_2, chi_B_vec=O_1, as in "
                             "CTINT, with the result <O_2(tau) O_1(0)> in chiAB_tau and chiAB_dlr")

        # -- Tail post proc flags

        perform_post_proc = params_kw.pop("perform_post_proc", True)
        perform_tail_fit = params_kw.pop("perform_tail_fit", False)
        dlr_sigma = params_kw.pop("dlr_sigma", None)

        if self.dlr:
            if perform_tail_fit:
                raise ValueError("perform_tail_fit does not apply in DLR mode: Sigma_iw is on the DLR mesh")
            dlr_sigma = dlr_sigma or 'dyson'
            if dlr_sigma not in ('dyson', 'crm'):
                raise ValueError(f"dlr_sigma must be 'dyson' or 'crm', got {dlr_sigma!r}")
            self._set_Delta_tau_from_dlr(params_kw)
        elif dlr_sigma is not None:
            raise ValueError("dlr_sigma needs the DLR mode (Solver(..., dlr_wmax=...))")

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

        # The moments come from this solve's density matrix, or are not known
        self.G_moments = self.Sigma_moments = self.Sigma_Hartree = self.orbital_occupations = None

        # chi_A_vec and chi_B_vec take two operators or two lists of them; the core takes lists, and chiAB_tau and
        # chiAB_dlr are scalar for two operators
        chi_ops = [params_kw.get(key) for key in ("chi_A_vec", "chi_B_vec")]
        self._chiAB_scalar = all(isinstance(O, Operator) for O in chi_ops)
        for key, O in zip(("chi_A_vec", "chi_B_vec"), chi_ops):
            if O is not None:
                params_kw[key] = [O] if isinstance(O, Operator) else list(O)

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

            if self.dlr:
                self._post_proc_dlr(dlr_sigma)
            else:
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

        if self.last_solve_parameters.measure_D0_corr and self.last_solve_parameters.measure_density_matrix:
            from triqs.operators import n
            from triqs.atom_diag import trace_rho_op

            def equal_time(op):
                return trace_rho_op(self.density_matrix, op, self.h_loc_diagonalization).real

            # Add back the equal-time part the kink estimator leaves out
            ops = self.conserved_density_operators
            offset = np.array([[equal_time(Oi * Oj) for Oj in ops] for Oi in ops])
            self.Q_conserved_l.data[0, :, :] += offset
            self.Q_conserved_tau.data[:] += offset
            if self.Q_tau is not None:
                for bl1, bl2 in self.Q_tau.indices:
                    n1, n2 = self.Q_tau[bl1, bl2].target_shape
                    offset = np.array([[equal_time(n(bl1, i1) * n(bl2, i2)) for i2 in range(n2)] for i1 in range(n1)])
                    self.Q_l[bl1, bl2].data[0, :, :] += offset
                    self.Q_tau[bl1, bl2].data[:] += offset
                # The C++ solver measures nn_tau by the sweep only where Q_tau cannot give it
                if self.last_solve_parameters.measure_nn_tau:
                    self.nn_tau = self.Q_tau.copy()

        return solve_status

    def _set_Delta_tau_from_dlr(self, params_kw):
        """DLR mode: sample Delta_iw on the n_tau points of the core's Delta_tau. With the G0_iw input, Delta_iw and
        h_loc0 (added to params_kw) are first computed from G0_iw: h_loc0 is the constant part of iw - G0^-1(iw),
        fitted on n_iw regular frequencies as the C++ G0_iw interface does, and Delta = iw - h_loc0 - G0^-1."""
        if not self.delta_interface:
            if 'h_loc0' in params_kw:
                raise ValueError("h_loc0 is computed from G0_iw: pass it only with delta_interface=True")
            p = SolveParametersT(**params_kw)
            h_loc0_mat = np.empty(len(self.gf_struct), dtype=object)
            for i, (bl, g0) in enumerate(self.G0_iw):
                g0_reg = make_gf_imfreq(make_gf_dlr(g0), n_iw=self.n_iw)
                x = g0_reg.copy()
                x << iOmega_n - inverse(g0_reg)
                tail, err = fit_hermitian_tail(x)
                if err > 1e-5 and mpi.is_master_node():
                    print(f"WARNING: Tail fit in extraction of h_loc0 from G0_iw[{bl}] has large error of: {err}")
                # An imaginary part within the fit's residual is roundoff
                h = tail[0].real.copy() if np.abs(tail[0].imag).max() <= max(p.imag_threshold, err) else tail[0].copy()
                h[(np.abs(h) < p.off_diag_threshold) & ~np.eye(len(h), dtype=bool)] = 0.0
                self.Delta_iw[bl] << iOmega_n - h - inverse(g0)
                h_loc0_mat[i] = h
            params_kw['h_loc0'] = op_from_block_matrix(h_loc0_mat, self.gf_struct)
        self.Delta_tau << make_gf_imtime(self.Delta_iw, self.n_tau)

    def _has_retarded_interaction(self):
        return self._retarded or np.abs(self.Jperp_tau.data).max() > 0 or any(
            np.abs(self.D0_tau[b1, b2].data).max() > 0 for b1, b2 in self.D0_tau.indices)

    def _fit_G_dlr(self):
        """The least-squares DLR fit of G_tau with its high-frequency moments held: the 1/iw one (the identity), and
        the 1/iw^2 one from the density matrix when it is known and there is no retarded interaction, which it would
        miss. A free fit misses them through the noisy ends of G(tau), and Sigma at the high DLR frequencies inherits
        that error times w."""
        mesh = MeshDLR(self.dlr_mesh)
        unit = Gf(mesh=mesh, target_shape=[])
        K = np.empty((self.n_tau, len(mesh)))  # G(tau) of each DLR coefficient alone: one exponential each
        for l in range(len(mesh)):
            unit.data[:] = 0
            unit.data[l] = 1
            K[:, l] = make_gf_imtime(unit, self.n_tau).data.real
        # G'(tau) = -w G(tau) for the coefficient at the DLR frequency w (MeshDLR holds beta w). Not -log(G(dtau) / G(0))
        # / dtau: for the nodes beyond beta w ~ -745 (beta w_max = 800 at beta 40, w_max 20) both underflow to 0
        rate = np.real(mesh.values()) / self.dlr_mesh.beta
        A = -(K[0] + K[-1])[None, :]  # -(G(0) + G(beta)), the 1/iw moment
        with_M2 = self.G_moments is not None and not self._has_retarded_interaction()
        if with_M2:
            A = np.vstack([A, rate * A])  # G'(0) + G'(beta), the 1/iw^2 moment
        # c = c_p + N z with A c_p = b and A N = 0, z by least squares
        N = np.linalg.svd(A)[2][len(A):].T
        KN_pinv = np.linalg.pinv(K @ N, rcond=self.dlr_mesh.eps)  # no directions below the DLR accuracy

        G_dlr = BlockGf(mesh=mesh, gf_struct=self.gf_struct)
        for bl, g in self.G_tau:
            moments = [np.eye(g.target_shape[0])] + ([self.G_moments[bl][2]] if with_M2 else [])
            for i, j in np.ndindex(*g.target_shape):
                c_p = np.linalg.lstsq(A, np.array([m[i, j] for m in moments]), rcond=None)[0]
                G_dlr[bl].data[:, i, j] = c_p + N @ (KN_pinv @ (g.data[:, i, j] - K @ c_p))
        return G_dlr

    def _post_proc_dlr(self, dlr_sigma):
        """DLR mode: G_iw from the DLR fit of G_tau, and Sigma_iw by Dyson on the DLR mesh (CRM or plain, which the CRM
        falls back to where it fails)."""
        self.G_dlr = self._fit_G_dlr()
        self.G_iw = make_gf_dlr_imfreq(self.G_dlr)
        self.G_tau_dlr = make_gf_dlr_imtime(self.G_dlr)

        if self.delta_interface:
            G0_iw = self.Delta_iw.copy()
            h_loc0_mat = block_matrix_from_op(self.h_loc0, self.gf_struct)
            for i, (bl, g0) in enumerate(G0_iw):
                g0 << inverse(iOmega_n - h_loc0_mat[i] - self.Delta_iw[bl])
        else:
            G0_iw = self.G0_iw

        self.crm_failure = None
        if dlr_sigma == 'crm':
            if self.Sigma_moments is None:
                raise ValueError("dlr_sigma='crm' needs the high-frequency moments of Sigma: set measure_density_matrix=True")
            from triqs.gfs.dlr_crm_dyson_solver import minimize_dyson
            Sigma_iw, failure = None, None
            if mpi.is_master_node():
                try:
                    Sigma_iw = G0_iw.copy()
                    np.random.seed(85281)
                    for bl, sigma in Sigma_iw:
                        sigma_dyn, _, _ = minimize_dyson(G0_dlr=G0_iw[bl], G_dlr=self.G_iw[bl], Sigma_moments=self.Sigma_moments[bl])
                        sigma << sigma_dyn
                        sigma += self.Sigma_moments[bl][0]
                    if not all(np.isfinite(sigma.data).all() for _, sigma in Sigma_iw):
                        raise FloatingPointError("non-finite Sigma")
                except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
                    Sigma_iw, failure = None, f"{type(e).__name__}: {e}"
                    print(f"WARNING: the CRM failed ({failure}); Sigma_iw is the plain Dyson G0^-1 - G^-1", flush=True)
            # The master decides: every rank takes the same branch
            Sigma_iw, self.crm_failure = mpi.bcast((Sigma_iw, failure))
            if self.crm_failure is None:
                self.Sigma_iw = Sigma_iw
                return

        self.Sigma_iw = G0_iw.copy()
        self.Sigma_iw << inverse(G0_iw) - inverse(self.G_iw)
