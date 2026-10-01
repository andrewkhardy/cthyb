# Two-patch DCA model with a *real-space* retarded spin-spin interaction, shared by run_cthyb.py,
# run_ed.py and calibrate_mu.py (and loaded by path by SYK_yukawa/DCA/dca_syk.py).
#
# Two-patch DCA (VBDMFT, Ferrero et al. PRB 80, 064501) of the 2D Hubbard model: the zone is cut
# into a central patch (|kx|, |ky| < pi/sqrt(2)) and the rest, and the solver's operators are
# the patch operators c_{K sigma}, K in {0, 1} = {even, odd}, in which the bath is diagonal. The
# two cluster sites are the rotated combinations c_{i sigma} = sum_K R[i, K] c_{K sigma},
# R = [[1, 1], [1, -1]] / sqrt(2). The interaction is local on the sites, hence off-diagonal in
# the working basis:
#
#     H = sum_{K sigma} (eps_K - mu) n_{K sigma}          patch levels + chemical potential
#       + U sum_i n_{i up} n_{i down}                     static Hubbard U, on the sites
#       + sum_{ij} lambda_ij(tau) S_i(tau) . S_j(0)       retarded S.S, on the sites
#
# with lambda_ij(tau) = -J_ij Q(tau), Q(tau) = -cosh(omega_0 (tau - beta/2)) / (2 omega_0 sinh(omega_0 beta/2)),
# J_ii = J_intra and J_ij = J_inter for i != j.
#
# solver.add_dyn_vertex(op1, op2, coupling) registers coupling(tau) op1(tau) op2(0) for single
# bilinear monomials op1, op2, and IGNORES their scalar coefficients. A site spin operator is a
# sum of patch bilinears, so expand_retarded_product multiplies op1 (x) op2 out term by term,
# keeps unit monomials and moves every numeric factor into the coupling, merging repeated pairs.
#
# Convention: S.S = S^z S^z + (1/2)(S^+ S^- + S^- S^+) with lambda = -J Q gives, with the site
# basis equal to the working basis (--rotation none), the single-orbital spin_spin benchmark's
# inputs: (n_up, n_up) -> -J/4 Q, (n_up, n_down) -> +J/4 Q, (S^+, S^-) -> -J/2 Q.

from itertools import product
import numpy as np
from triqs.gfs import BlockGf, GfImFreq, GfImTime, inverse, iOmega_n
from triqs.operators import c, c_dag

SPIN_NAMES = ('up', 'down')
N_PATCH = 2

# c_{i sigma} = sum_K ROTATION[i, K] c_{K sigma}; orthogonal and symmetric.
ROTATION = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
IDENTITY = np.eye(2)


def _monomial_key(monomial):
    """Hashable key for one TRIQS monomial: ((dagger, (indices...)), ...)."""
    return tuple((bool(dagger), tuple(indices)) for dagger, indices in monomial)


def key_to_operator(key):
    """The monomial as a TRIQS Operator with unit coefficient (add_dyn_vertex drops any other)."""
    op = None
    for dagger, indices in key:
        factor = c_dag(*indices) if dagger else c(*indices)
        op = factor if op is None else op * factor
    return op


def key_to_string(key):
    """Stable text form of a monomial key, e.g. "c_dag(up,0)*c(down,1)", to label the measured
    vertex correlators."""
    return '*'.join(f"{'c_dag' if dagger else 'c'}({','.join(str(i) for i in indices)})"
                    for dagger, indices in key)


def expand_retarded_product(op1, op2, prefactor=1.0, vertices=None):
    """Expand prefactor * op1(tau) op2(0) into `{(monomial1, monomial2): coefficient}`, merging
    repeated pairs. op1 and op2 are never multiplied as operators: they sit at different times."""
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


def add_model_args(parser):
    parser.add_argument('--beta', type=float, default=10.0, help='Inverse temperature')
    parser.add_argument('--t', type=float, default=0.25, help='Nearest-neighbour hopping (bandwidth 8t)')
    parser.add_argument('--tp', type=float, default=0.0,
                        help="Next-nearest hopping t'; 0 keeps particle-hole symmetry")
    parser.add_argument('--U', type=float, default=2.0, help='Static Hubbard U, local on the two cluster sites')
    parser.add_argument('--J_intra', type=float, default=0.0,
                        help='Retarded on-site S.S coupling J_ii (coupling is -J_ii Q(tau))')
    parser.add_argument('--J_inter', type=float, default=0.5, help='Retarded inter-site S.S coupling J_ij, i != j')
    parser.add_argument('--omega_0', type=float, default=1.0, help='Boson frequency setting Q(tau)')
    parser.add_argument('--mu', type=float, default=None, help='Chemical potential; default U/2')
    parser.add_argument('--bath', choices=['dca', 'discrete'], default='dca',
                        help="'dca': coarse-grained 2D dispersion; 'discrete': one bath site per patch "
                             "and spin, V^2/(iw - eps_bath), ED-representable")
    parser.add_argument('--V', type=float, default=0.5, help='Hybridization strength for --bath discrete')
    parser.add_argument('--eps_bath', type=float, default=0.0, help='Bath-site energy for --bath discrete')
    parser.add_argument('--rotation', choices=['site', 'none'], default='site',
                        help="'site': the interaction is local in the rotated basis (the DCA case); "
                             "'none': local in the working basis")
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

        # One block per spin, the two patches as the orbital index, so the interaction's
        # patch-off-diagonal bilinears live inside a block.
        self.gf_struct = [(s, N_PATCH) for s in SPIN_NAMES]
        self.labels = list(product(SPIN_NAMES, range(N_PATCH)))  # CTHYB's linear-index order

        self.eps_patch, self.rho_patch = self._coarse_grain()

        # A pure spin-spin retarded interaction carries no static charge shift: mu = U/2 is the
        # particle-hole symmetric value.
        self.mu = 0.5 * self.U if args.mu is None else args.mu

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

        # The same in the working (patch) basis
        self.Sz_patch = [0.5 * (c_dag('up', K) * c('up', K) - c_dag('down', K) * c('down', K)) for K in range(N_PATCH)]
        self.Sp_patch = [c_dag('up', K) * c('down', K) for K in range(N_PATCH)]
        self.Sm_patch = [c_dag('down', K) * c('up', K) for K in range(N_PATCH)]

        # Total S^z, the same in either basis (R is orthogonal); it commutes with h_loc.
        self.Sz_total = sum(self.Sz_patch)

    def _coarse_grain(self):
        """Patch-averaged level eps_K and partial density of states rho_K(eps), from the 2D
        dispersion cut at |kx|, |ky| < pi/sqrt(2)."""
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

        # Each patch is exactly half the zone, so eps_even = -eps_odd exactly for any t, t'. The
        # finite k-grid misses that; subtract the mean (a choice of energy zero) and keep it.
        self.eps_asymmetry = 0.5 * (eps_patch[0] + eps_patch[1])
        eps_patch = [e - self.eps_asymmetry for e in eps_patch]

        # The two patch densities of states are not mirror images, so the dca bath breaks
        # particle-hole symmetry and mu = U/2 is not exactly half filling there.
        self.dos_variance_ratio = float(
            np.sum(rho_patch[0][1] * (rho_patch[0][0] - eps_patch[0] - self.eps_asymmetry)**2)
            / np.sum(rho_patch[1][1] * (rho_patch[1][0] - eps_patch[1] - self.eps_asymmetry)**2))
        return eps_patch, rho_patch

    def delta_iw(self, n_iw):
        """Hybridization Delta_K(iw), diagonal in the patch index, without the patch level eps_K
        (which belongs in h_loc0, so that Delta(tau) decays)."""
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

    def h_int(self):
        """Static Hubbard U, local on the two cluster sites (off-diagonal in the patch basis)."""
        return self.U * sum(self.n_site[(i, 'up')] * self.n_site[(i, 'down')] for i in range(N_PATCH))

    def h_loc0(self):
        """Patch levels and chemical potential; diagonal in the working basis."""
        return sum((self.eps_patch[K] - self.mu) * c_dag(s, K) * c(s, K)
                   for s in SPIN_NAMES for K in range(N_PATCH))

    def half_filling_is_exact(self):
        """True if mu = U/2 is exactly half filling: the interaction is particle-hole even and the
        patch levels antisymmetric, so only the bath can spoil it, and the dca one does."""
        return self.bath == 'discrete'

    def Q(self, tau):
        return -np.cosh(self.omega_0 * (tau - self.beta / 2)) / (2 * self.omega_0 * np.sinh(self.omega_0 * self.beta / 2))

    def Q_tau(self, n_tau_bosonic):
        Q = GfImTime(target_shape=(1, 1), beta=self.beta, statistic='Boson', n_points=n_tau_bosonic)
        Q.data[:, 0, 0] = self.Q(np.linspace(0, self.beta, n_tau_bosonic))
        return Q

    def spin_spin_vertices(self, basis='site'):
        """Merged monomial-pair coefficients (multiplying Q(tau)) of sum_ij -J_ij Q(tau) S_i(tau).S_j(0),
        i, j over the sites (basis='site') or patches ('patch'). Every ordered (i, j) is expanded
        separately, so the list is closed under swapping op1 and op2, as the moves need."""
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
        """Hand spin_spin_vertices() to the solver; returns the (op1, op2, coefficient) registered."""
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
