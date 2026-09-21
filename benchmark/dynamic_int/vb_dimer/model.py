# Shared model definition for the two-patch DCA benchmark with a *real-space* retarded
# spin-spin interaction. Used by check_rotation.py (symbolic checks, no MC),
# run_cthyb.py (the QMC run) and run_ed.py (the ED reference) -- so every driver
# solves exactly the same problem.
#
# ---------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------
# Two-patch DCA (VBDMFT, Ferrero et al. PRB 80, 064501) of the 2D Hubbard model. The
# Brillouin zone is cut into a central patch P+ (|kx|, |ky| < pi/sqrt(2)) and the rest P-,
# so the impurity has two orbitals -- the two patch-averaged Fermi-surface points. The
# solver's fundamental operators are the *patch* operators c_{K sigma}, K in {0, 1} =
# {even, odd}, because the bath is diagonal in K. The two effective cluster "sites" are
# the rotated combinations
#
#     c_{1 sigma} = (c_{even sigma} + c_{odd sigma}) / sqrt(2)
#     c_{2 sigma} = (c_{even sigma} - c_{odd sigma}) / sqrt(2)
#
# i.e. c_{i sigma} = sum_K R[i, K] c_{K sigma} with R = [[1, 1], [1, -1]] / sqrt(2), the
# same construction as benchmark/dynamic_int/../vbdmft.py. The interaction is local in the
# *site* basis and therefore off-diagonal in the solver's working basis:
#
#     H = sum_{K sigma} (eps_K - mu) n_{K sigma}          patch levels + chemical potential
#       + U sum_i n_{i up} n_{i down}                     static Hubbard U, on the sites
#       + sum_{ij} lambda_ij(tau) S_i(tau) . S_j(0)       retarded S.S, on the sites
#
# with lambda_ij(tau) = -J_ij Q(tau) and Q(tau) the single-boson kernel used throughout
# benchmark/dynamic_int,
#
#     Q(tau) = -cosh(omega_0 (tau - beta/2)) / (2 omega_0 sinh(omega_0 beta / 2)).
#
# J_ij is J_intra on the diagonal and J_inter off it, so J_intra == J_inter is the
# special "uniform" case where sum_{ij} S_i.S_j collapses to S_tot.S_tot (see below).
#
# ---------------------------------------------------------------------------------------
# Why the expansion below is needed
# ---------------------------------------------------------------------------------------
# solver.add_dyn_vertex(op1, op2, coupling) registers the retarded term
# coupling(tau) * op1(tau) op2(0). It requires op1 and op2 to each be a *single* fermion
# bilinear monomial (extract_bilinear in c++/triqs_cthyb/dynamical_interactions.cpp), and
# -- important -- it *ignores* the scalar coefficient of op1/op2 entirely. A site-basis
# spin operator is a sum of patch-basis bilinears,
#
#     S_i^z = (1/2) sum_{K K'} R[i,K] R[i,K'] (c^dag_{K up} c_{K' up} - c^dag_{K dn} c_{K' dn})
#     S_i^+ =       sum_{K K'} R[i,K] R[i,K'] c^dag_{K up} c_{K' dn}
#
# so it cannot be handed to add_dyn_vertex directly. expand_retarded_product below does
# the expansion explicitly: it multiplies out op1 (x) op2 term by term, keeps the
# monomials as operators and accumulates every numeric factor into the *coupling*, which
# is the only place a prefactor survives. Monomial pairs that appear more than once are
# merged (their coefficients added) rather than registered twice.
#
# ---------------------------------------------------------------------------------------
# Convention anchor
# ---------------------------------------------------------------------------------------
# lambda(tau) S(tau).S(0) with S.S = S^z S^z + (1/2)(S^+ S^- + S^- S^+) and
# lambda = -J Q reproduces benchmark/dynamic_int/spin_spin.py -- the validated
# single-orbital full-S.S benchmark -- coefficient for coefficient:
#
#     (n_up,   n_up  ) -> -J/4 Q   == spin_spin.py's D0_tau["up",  "up"  ] = -0.25 J Q
#     (n_up,   n_down) -> +J/4 Q   == spin_spin.py's D0_tau["up",  "down"] = +0.25 J Q
#     (S^+,    S^-   ) -> -J/2 Q   == spin_spin.py's Jperp_tau / 2 = -J Q / 2
#
# check_rotation.py asserts this numerically (with --rotation none, where the site basis
# is the working basis), so the convention is pinned to the validated script rather than
# re-derived here.

from itertools import product
import numpy as np
from triqs.gfs import BlockGf, GfImFreq, GfImTime, inverse, iOmega_n
from triqs.operators import c, c_dag, Operator

SPIN_NAMES = ('up', 'down')
N_PATCH = 2
PATCH_NAMES = ('even', 'odd')

# c_{i sigma} = sum_K ROTATION[i, K] c_{K sigma}; orthogonal and symmetric.
ROTATION = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
IDENTITY = np.eye(2)


# ---------------------------------------------------------------------------------------
# Monomial expansion for add_dyn_vertex
# ---------------------------------------------------------------------------------------

def _monomial_key(monomial):
    """Hashable key for one TRIQS monomial: ((dagger, (indices...)), ...)."""
    return tuple((bool(dagger), tuple(indices)) for dagger, indices in monomial)


def key_to_operator(key):
    """Rebuild the monomial as a TRIQS Operator with unit coefficient -- add_dyn_vertex
    drops any coefficient on its operator arguments, so it must be exactly 1 here and
    the real factor must travel in the coupling instead."""
    op = None
    for dagger, indices in key:
        factor = c_dag(*indices) if dagger else c(*indices)
        op = factor if op is None else op * factor
    return op


def key_to_string(key):
    """Stable text form of a monomial key, e.g. "c_dag(up,0)*c(down,1)". Used to label the
    measured vertex correlators in the output file so the analysis script can match them
    back to the expansion coefficients without relying on ordering."""
    return '*'.join(f"{'c_dag' if dagger else 'c'}({','.join(str(i) for i in indices)})"
                    for dagger, indices in key)


def expand_retarded_product(op1, op2, prefactor=1.0, vertices=None):
    """Expand prefactor * op1(tau) op2(0) into single-bilinear monomial pairs.

    op1 and op2 are arbitrary sums of fermion bilinears (e.g. site-basis spin operators
    written in the patch basis). Returns/updates a dict keyed by the pair of monomials,
    whose value is the accumulated numeric coefficient. Note the two operators are *never*
    multiplied together as operators: they sit at different times, so normal-ordering
    across them would be wrong. Each is expanded on its own and the monomials paired.
    """
    if vertices is None:
        vertices = {}
    for mono1, coeff1 in op1:
        key1 = _monomial_key(mono1)
        if len(key1) != 2 or key1[0][0] == key1[1][0]:
            raise ValueError(f"op1 contains the non-bilinear monomial {key1}; "
                             "add_dyn_vertex needs one creation and one annihilation operator per term")
        for mono2, coeff2 in op2:
            key2 = _monomial_key(mono2)
            if len(key2) != 2 or key2[0][0] == key2[1][0]:
                raise ValueError(f"op2 contains the non-bilinear monomial {key2}; "
                                 "add_dyn_vertex needs one creation and one annihilation operator per term")
            coeff = prefactor * coeff1 * coeff2
            if abs(np.imag(coeff)) > 1e-12:
                raise ValueError(f"Complex vertex coefficient {coeff} for {key1} x {key2}; "
                                 "the real rotation used here should never produce one")
            vertices[(key1, key2)] = vertices.get((key1, key2), 0.0) + np.real(coeff)
    return vertices


def prune(vertices, tol=1e-12):
    """Drop monomial pairs whose merged coefficient cancelled to zero."""
    return {key: val for key, val in vertices.items() if abs(val) > tol}


# ---------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------

def ph_swap_transform(op):
    """Particle-hole transform combined with the even <-> odd patch swap: c_{K s} -> c^dag_{Kbar s}.

    The swap is needed because the two patch levels are exactly opposite (eps_even = -eps_odd):
    plain particle-hole flips the sign of the level term and the swap flips it back. On the
    cluster sites this acts as c_{i s} -> +- c^dag_{i s}, so n_{i s} -> 1 - n_{i s} and the
    site-local Hubbard U maps onto itself. Each factor is mapped in place and TRIQS re-canonicalises
    the monomial, which generates the anticommutator signs and lower-order terms automatically.
    """
    out = Operator()
    for monomial, coeff in op:
        term = None
        for dagger, indices in monomial:
            spin, patch = indices[0], 1 - indices[1]
            factor = c(spin, patch) if dagger else c_dag(spin, patch)
            term = factor if term is None else term * factor
        out += coeff * (1.0 if term is None else term)
    return out


def add_model_args(parser):
    parser.add_argument('--beta', type=float, default=10.0, help='Inverse temperature')
    parser.add_argument('--t', type=float, default=0.25, help='Nearest-neighbour hopping (bandwidth 8t)')
    parser.add_argument('--tp', type=float, default=0.0,
                        help="Next-nearest hopping t'; VBDMFT uses -0.3t, 0 keeps particle-hole symmetry")
    parser.add_argument('--U', type=float, default=2.0, help='Static Hubbard U, local on the two cluster sites')
    parser.add_argument('--J_intra', type=float, default=0.0,
                        help='Retarded on-site S.S coupling J_ii (coupling is -J_ii Q(tau))')
    parser.add_argument('--J_inter', type=float, default=0.5,
                        help='Retarded inter-site S.S coupling J_ij, i != j -- the DCA channel of interest')
    parser.add_argument('--omega_0', type=float, default=1.0, help='Boson frequency setting Q(tau)')
    parser.add_argument('--mu', type=float, default=None,
                        help='Chemical potential; default U/2 (the particle-hole symmetric value, since a pure '
                             'spin-spin retarded interaction adds no static charge shift of its own)')
    parser.add_argument('--bath', choices=['dca', 'discrete'], default='dca',
                        help="'dca': hybridization from coarse-graining the 2D dispersion over the two patches. "
                             "'discrete': one bath site per patch and spin, V^2/(iw - eps_bath) -- not the DCA "
                             "bath, but exactly representable in ED, so this is the variant an ED reference uses")
    parser.add_argument('--V', type=float, default=0.5, help='Hybridization strength for --bath discrete')
    parser.add_argument('--eps_bath', type=float, default=0.0, help='Bath-site energy for --bath discrete')
    parser.add_argument('--rotation', choices=['site', 'none'], default='site',
                        help="'site': the interaction is local in the rotated (real-space) basis -- the DCA case. "
                             "'none': the rotation is the identity, so the interaction is local in the working "
                             "basis; used by check_rotation.py to pin conventions against spin_spin.py")
    parser.add_argument('--n_k', type=int, default=1000, help='Linear k-grid size for the coarse-graining')
    parser.add_argument('--n_bins', type=int, default=50, help='Energy bins for each patch density of states')


class Model:

    def __init__(self, args):
        self.beta, self.t, self.tp, self.U = args.beta, args.t, args.tp, args.U
        self.J_intra, self.J_inter = args.J_intra, args.J_inter
        self.omega_0 = args.omega_0
        self.bath, self.V, self.eps_bath = args.bath, args.V, args.eps_bath
        self.rotation_name = args.rotation
        self.R = ROTATION if args.rotation == 'site' else IDENTITY
        self.n_k, self.n_bins = args.n_k, args.n_bins

        # gf_struct: one block per spin, the two patches as the orbital index. The
        # interaction's patch-off-diagonal bilinears then live inside a block.
        self.gf_struct = [(s, N_PATCH) for s in SPIN_NAMES]
        self.labels = list(product(SPIN_NAMES, range(N_PATCH)))  # CTHYB's linear-index order

        self.eps_patch, self.rho_patch = self._coarse_grain()

        # A pure spin-spin retarded interaction carries no static charge shift, so the
        # particle-hole symmetric value is just U/2 (as in spin_spin.py). The solver's
        # own Lang-Firsov bookkeeping is reported by the driver so this can be checked.
        self.mu_is_default = args.mu is None
        self.mu = 0.5 * self.U if self.mu_is_default else args.mu

        # Site-basis operators, written in the patch basis
        self.c_site = {(i, s): sum(self.R[i, K] * c(s, K) for K in range(N_PATCH))
                       for i in range(N_PATCH) for s in SPIN_NAMES}
        self.c_dag_site = {(i, s): sum(self.R[i, K] * c_dag(s, K) for K in range(N_PATCH))
                           for i in range(N_PATCH) for s in SPIN_NAMES}
        self.n_site = {(i, s): self.c_dag_site[(i, s)] * self.c_site[(i, s)]
                       for i in range(N_PATCH) for s in SPIN_NAMES}
        self.Sz_site = [0.5 * (self.n_site[(i, 'up')] - self.n_site[(i, 'down')]) for i in range(N_PATCH)]
        self.Sp_site = [self.c_dag_site[(i, 'up')] * self.c_site[(i, 'down')] for i in range(N_PATCH)]
        self.Sm_site = [self.c_dag_site[(i, 'down')] * self.c_site[(i, 'up')] for i in range(N_PATCH)]

        # The same objects in the working (patch) basis, for the basis-independence check
        self.Sz_patch = [0.5 * (c_dag('up', K) * c('up', K) - c_dag('down', K) * c('down', K)) for K in range(N_PATCH)]
        self.Sp_patch = [c_dag('up', K) * c('down', K) for K in range(N_PATCH)]
        self.Sm_patch = [c_dag('down', K) * c('up', K) for K in range(N_PATCH)]

        # Total spin: basis independent, since sum_i c^dag_{i s} c_{i s'} is the trace of
        # c^dag c over the site index and R is orthogonal.
        self.Sz_total = sum(self.Sz_patch)
        self.Sp_total = sum(self.Sp_patch)
        self.Sm_total = sum(self.Sm_patch)
        self.N_total = sum(c_dag(s, K) * c(s, K) for s in SPIN_NAMES for K in range(N_PATCH))

    # -----------------------------------------------------------------------------------
    # Lattice / bath
    # -----------------------------------------------------------------------------------

    def _coarse_grain(self):
        """Patch-averaged level eps_K and partial density of states rho_K(eps), from the
        2D dispersion cut at |kx|, |ky| < pi/sqrt(2) (vbdmft.py's patches)."""
        k = np.linspace(-np.pi, np.pi, self.n_k)
        kx, ky = np.meshgrid(k, k)
        epsk = -2 * self.t * (np.cos(kx) + np.cos(ky)) - 4 * self.tp * np.cos(kx) * np.cos(ky)
        central = (np.abs(kx) < np.pi / np.sqrt(2)) & (np.abs(ky) < np.pi / np.sqrt(2))

        eps_patch, rho_patch = [], []
        for mask in (central, np.invert(central)):
            counts, edges = np.histogram(np.extract(mask, epsk), bins=self.n_bins, density=True)
            centres = 0.5 * (edges[:-1] + edges[1:])
            weights = counts * (edges[1] - edges[0])  # sums to 1 over the patch
            eps_patch.append(float(np.sum(weights * centres)))
            rho_patch.append((centres, weights))

        # eps_even = -eps_odd EXACTLY, for any t and t'. The boundary a = pi/sqrt(2) is chosen so
        # the central patch has area 4a^2 = 2 pi^2, exactly half the Brillouin zone -- that is what
        # picks pi/sqrt(2). Both cos kx + cos ky and cos kx cos ky average to zero over the full
        # zone, so A_+ eps_+ + A_- eps_- = int_BZ eps = 0 with A_+ = A_-, hence eps_+ = -eps_-.
        # (Closed form for t' = 0: eps_+ = -4 t sin(a)/a.) Whatever the finite k-grid produces
        # instead is discretization error -- it tracks the patch's area fraction, which is 0.5 only
        # up to how many grid points land inside the mask, so it does NOT shrink monotonically with
        # n_k (measured: +1.8e-3, -2.2e-4, +3.7e-5 at n_k = 1000, 4000, 16000). An exact structural
        # property is imposed, not hoped for: subtract the mean, and keep it for reporting. This is
        # a pure choice of energy zero, so nothing else has to change.
        self.eps_asymmetry = 0.5 * (eps_patch[0] + eps_patch[1])
        eps_patch = [e - self.eps_asymmetry for e in eps_patch]

        # The two patch densities of states are NOT mirror images: their variances differ by ~18%.
        # That is real -- the patches are physically different -- and it means the coarse-grained
        # (--bath dca) hybridization breaks particle-hole symmetry, so mu = U/2 is not exactly half
        # filling there. See half_filling_is_exact().
        self.dos_variance_ratio = float(
            np.sum(rho_patch[0][1] * (rho_patch[0][0] - eps_patch[0] - self.eps_asymmetry)**2)
            / np.sum(rho_patch[1][1] * (rho_patch[1][0] - eps_patch[1] - self.eps_asymmetry)**2))
        return eps_patch, rho_patch

    def delta_iw(self, n_iw):
        """Hybridization Delta_K(iw), diagonal in the patch index, with the patch level
        eps_K removed (it belongs in h_loc0, so that Delta(tau) decays as it must)."""
        delta = BlockGf(name_list=list(SPIN_NAMES),
                        block_list=[GfImFreq(beta=self.beta, n_points=n_iw, target_shape=(N_PATCH, N_PATCH))
                                    for _ in SPIN_NAMES])
        for _, d in delta:
            d.zero()
            for K in range(N_PATCH):
                if self.bath == 'discrete':
                    d[K, K] << self.V**2 * inverse(iOmega_n - self.eps_bath)
                else:
                    # Coarse-grained patch Green function at Sigma = 0, then
                    # Delta_K = iw - eps_K - 1/G_K: a fixed bath, independent of mu.
                    G_K = GfImFreq(beta=self.beta, n_points=n_iw, target_shape=(1, 1))
                    G_K.zero()
                    centres, weights = self.rho_patch[K]
                    for eps, w in zip(centres, weights):
                        G_K << G_K + w * inverse(iOmega_n - eps)
                    d[K, K] << (iOmega_n - self.eps_patch[K] - inverse(G_K)[0, 0])
        return delta

    # -----------------------------------------------------------------------------------
    # Hamiltonian
    # -----------------------------------------------------------------------------------

    def h_int(self):
        """Static Hubbard U, local on the two cluster sites (off-diagonal in the patch basis)."""
        return self.U * sum(self.n_site[(i, 'up')] * self.n_site[(i, 'down')] for i in range(N_PATCH))

    def h_loc0(self):
        """Patch levels and chemical potential; diagonal in the working basis."""
        return sum((self.eps_patch[K] - self.mu) * c_dag(s, K) * c(s, K)
                   for s in SPIN_NAMES for K in range(N_PATCH))

    def h_loc(self):
        return self.h_int() + self.h_loc0()

    def Kprime_0(self, coeff=1.0):
        r"""K'(0) for a vertex whose coupling is `coeff * Q(tau)` -- the *instantaneous* content of
        the retarded interaction, i.e. the constant offset that has to be moved into h_loc.

        Computed in closed form, with no quadrature, no Legendre fit and no solver run:

        K is defined by K'' = D, K(0) = K(beta) = 0, so (Fubini on the double integral)
            K'(0) = -(1/beta) int_0^beta (beta - tau) D(tau) dtau.
        Because (beta - tau) is degree 1 in x = 2 tau/beta - 1, only the l = 0 and l = 1 Legendre
        moments of D can contribute -- P_{l>=2} are orthogonal to every degree-1 polynomial. So
        cthyb's two-term formula K'(0) = -(beta/2)(d_0 - d_1/3) is an *identity*, not a truncation,
        and `dyn_n_l` cannot change the offset (verified in check_rotation.py).

        For this Q(tau) the two moments are elementary: int_0^beta Q = -1/omega_0^2, and
        int_0^beta Q(tau) x(tau) = 0 by symmetry about beta/2. Hence

            K'(0) = coeff / (2 omega_0^2).

        Cross-check against kanamori_phonon/model.py, whose phonon has D_ab = g_a g_b Q: the
        unordered-pair static shift -2K'(0) = -g_a g_b/omega_0^2 and the level shift
        -K'(0) = -g_a^2/(2 omega_0^2) are exactly the instantaneous part documented there.
        """
        return coeff / (2.0 * self.omega_0**2)

    def static_shift_operator(self, basis='site'):
        r"""The full instantaneous content of the retarded interaction, as an operator:
        sum_v K'_v(0) op1_v op2_v over *every* registered vertex.

        This is the quantity that matters for the filling, and it is a property of the input
        vertex list alone -- not of how the solver chooses to route each vertex. That is why it
        is computed here rather than read back from the solver: a route-dependent offset would
        give an lang_firsov=True run and an lang_firsov=False run different chemical potentials.
        See triqs_cthyb.dynamical_interactions.{static_shift, half_filling_mu} for the general
        density-coupled case, and pass verbosity=4 to solve() for the solver's routing audit.

        By the sum rule checked in check_rotation.py, sum_v coeff_v op1_v op2_v is exactly
        sum_ij (-J_ij) S_i.S_j, so here this reduces to
        (1/(2 omega_0^2)) sum_ij (-J_ij) S_i.S_j -- a pure *spin* bilinear, hence particle-hole
        even, hence contributing exactly nothing to the half-filling condition.
        """
        shift = Operator()
        for (key1, key2), coeff in self.spin_spin_vertices(basis).items():
            shift += self.Kprime_0(coeff) * key_to_operator(key1) * key_to_operator(key2)
        return shift

    def half_filling_is_exact(self):
        """True if mu = U/2 is exactly half filling.

        The interaction is particle-hole even (site-basis U, plus a pure spin-bilinear static
        offset) and the patch levels are exactly antisymmetric, so the only thing that can spoil
        it is the bath. A discrete bath with one identical site per patch has Delta_even =
        Delta_odd and is symmetric; the coarse-grained DCA bath is not (the two patch densities of
        states have different widths), so there mu must be tuned against the measured filling --
        which is exactly why vbdmft.py carries a doping -> mu table instead of using U/2.
        """
        return self.bath == 'discrete'

    def Q(self, tau):
        return -np.cosh(self.omega_0 * (tau - self.beta / 2)) / (2 * self.omega_0 * np.sinh(self.omega_0 * self.beta / 2))

    def Q_tau(self, n_tau_bosonic):
        Q = GfImTime(target_shape=(1, 1), beta=self.beta, statistic='Boson', n_points=n_tau_bosonic)
        Q.data[:, 0, 0] = self.Q(np.linspace(0, self.beta, n_tau_bosonic))
        return Q

    # -----------------------------------------------------------------------------------
    # The retarded S.S vertices
    # -----------------------------------------------------------------------------------

    def spin_spin_vertices(self, basis='site'):
        """Merged monomial-pair coefficients for sum_{ij} lambda_ij(tau) S_i(tau).S_j(0),
        with lambda_ij = -J_ij Q(tau). The returned coefficients multiply Q(tau).

        basis='site'  : i, j run over the two rotated cluster sites (the DCA case).
        basis='patch' : i, j run over the two patches instead -- used only by
                        check_rotation.py, where the uniform case must give the identical
                        vertex list because sum_i S_i = sum_K S_K.

        Every ordered pair (i, j) is registered separately: op1(tau) op2(0) and
        op2(tau) op1(0) are different retarded correlators, not one term counted twice.
        This is the same convention as kanamori_dynamical_vertices, and it makes the
        vertex list closed under swapping op1 and op2, which the moves rely on.
        """
        Sz = self.Sz_site if basis == 'site' else self.Sz_patch
        Sp = self.Sp_site if basis == 'site' else self.Sp_patch
        Sm = self.Sm_site if basis == 'site' else self.Sm_patch

        vertices = {}
        for i, j in product(range(N_PATCH), range(N_PATCH)):
            J_ij = self.J_intra if i == j else self.J_inter
            if J_ij == 0.0:
                continue
            # S_i . S_j = S_i^z S_j^z + (1/2)(S_i^+ S_j^- + S_i^- S_j^+), times -J_ij
            expand_retarded_product(Sz[i], Sz[j], -J_ij, vertices)
            expand_retarded_product(Sp[i], Sm[j], -0.5 * J_ij, vertices)
            expand_retarded_product(Sm[i], Sp[j], -0.5 * J_ij, vertices)
        return prune(vertices)

    def register_vertices(self, solver, n_tau_bosonic, basis='site'):
        """Hand spin_spin_vertices() to the solver. Returns the list of
        (op1, op2, coefficient) actually registered, for reporting and for the
        coupling-derivative reconstruction in the analysis script."""
        from triqs_cthyb.dynamical_interactions import _as_scalar_gf
        Q = self.Q_tau(n_tau_bosonic)
        registered = []
        for (key1, key2), coeff in sorted(self.spin_spin_vertices(basis).items()):
            op1, op2 = key_to_operator(key1), key_to_operator(key2)
            coupling = Q.copy()
            coupling.data[:] *= coeff
            solver.add_dyn_vertex(op1, op2, _as_scalar_gf(coupling))
            registered.append((op1, op2, coeff))
        return registered

    # -----------------------------------------------------------------------------------

    def params(self):
        return dict(beta=self.beta, t=self.t, tp=self.tp, U=self.U, J_intra=self.J_intra, J_inter=self.J_inter,
                    omega_0=self.omega_0, mu=self.mu, bath=self.bath, V=self.V, eps_bath=self.eps_bath,
                    rotation=self.rotation_name, eps_patch=np.array(self.eps_patch),
                    eps_asymmetry=self.eps_asymmetry, dos_variance_ratio=self.dos_variance_ratio)

    def tag(self):
        bath = f'V-{self.V}' if self.bath == 'discrete' else 'dca'
        return (f'beta-{self.beta}_t-{self.t}_tp-{self.tp}_U-{self.U}'
                f'_Jintra-{self.J_intra}_Jinter-{self.J_inter}_w0-{self.omega_0}'
                f'_mu-{self.mu}_bath-{bath}_rot-{self.rotation_name}')
