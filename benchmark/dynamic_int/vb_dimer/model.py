# Two-patch DCA model (VBDMFT, Ferrero et al. PRB 80, 064501) with a retarded spin-spin
# interaction between the two cluster sites. Shared by run_cthyb.py, run_ed.py and
# calibrate_mu.py.
#
# The solver works in the patch basis K in {even, odd}, where the bath is diagonal. The two
# cluster sites are the rotated combinations c_{i s} = sum_K R[i, K] c_{K s}, with
# R = [[1, 1], [1, -1]] / sqrt(2). The interactions are local on the sites:
#
#     H     = sum_{K s} (eps_K - mu) n_{K s} + U sum_i n_{i up} n_{i down}
#     S_dyn = 1/2 sum_ij int int -J_ij Q(tau - tau') S_i(tau) . S_j(tau')
#
# with Q(tau) = -cosh(omega_0 (tau - beta/2)) / (2 omega_0 sinh(omega_0 beta/2)), the kernel of
# one boson of frequency omega_0, and J_ij = J_intra (i == j) or J_inter (i != j).

from itertools import product
import numpy as np
from triqs.gfs import BlockGf, GfImFreq, inverse, iOmega_n
from triqs.operators import c, c_dag

SPIN_NAMES = ('up', 'down')
N_PATCH = 2  # also the number of sites

ROTATION = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)


def add_model_args(parser):
    parser.add_argument('--beta', type=float, default=10.0, help='Inverse temperature')
    parser.add_argument('--t', type=float, default=0.25, help='Nearest-neighbour hopping')
    parser.add_argument('--tp', type=float, default=0.0, help="Next-nearest hopping t'")
    parser.add_argument('--U', type=float, default=2.0, help='Hubbard U on each site')
    parser.add_argument('--J_intra', type=float, default=0.0, help='Retarded S_i.S_i coupling')
    parser.add_argument('--J_inter', type=float, default=0.5, help='Retarded S_1.S_2 coupling')
    parser.add_argument('--omega_0', type=float, default=1.0, help='Boson frequency in Q(tau)')
    parser.add_argument('--mu', type=float, default=None, help='Chemical potential, default U/2')
    parser.add_argument('--bath', choices=['dca', 'discrete'], default='dca',
                        help="'dca': coarse-grained 2D lattice. 'discrete': one bath site per patch "
                             "and spin, V^2/(iw - eps_bath), which ED can solve")
    parser.add_argument('--V', type=float, default=0.5, help='Hybridization for --bath discrete')
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
        self.J = np.array([[self.J_intra, self.J_inter], [self.J_inter, self.J_intra]])
        self.omega_0 = args.omega_0
        self.bath, self.V, self.eps_bath = args.bath, args.V, args.eps_bath
        self.rotation_name = args.rotation
        self.R = ROTATION if args.rotation == 'site' else np.eye(N_PATCH)
        self.n_k, self.n_bins = args.n_k, args.n_bins

        self.gf_struct = [(s, N_PATCH) for s in SPIN_NAMES]
        self.labels = list(product(SPIN_NAMES, range(N_PATCH)))

        self.eps_patch, self.rho_patch = self._coarse_grain()

        # The spin-spin interaction has no charge part, so mu = U/2 is half filling for a
        # particle-hole symmetric bath (see half_filling_is_exact).
        self.mu = 0.5 * self.U if args.mu is None else args.mu

        # Site spin operators, written in the patch basis
        self.Sz = [0.5 * (self.n_site(i, 'up') - self.n_site(i, 'down')) for i in range(N_PATCH)]
        self.Sp = [self.c_dag_site(i, 'up') * self.c_site(i, 'down') for i in range(N_PATCH)]
        self.Sm = [self.c_dag_site(i, 'down') * self.c_site(i, 'up') for i in range(N_PATCH)]
        self.Sz_total = sum(self.Sz)

    def c_site(self, i, s):
        return sum(self.R[i, K] * c(s, K) for K in range(N_PATCH))

    def c_dag_site(self, i, s):
        return sum(self.R[i, K] * c_dag(s, K) for K in range(N_PATCH))

    def n_site(self, i, s):
        return self.c_dag_site(i, s) * self.c_site(i, s)

    # -----------------------------------------------------------------------------------

    def _coarse_grain(self):
        """Patch level eps_K and density of states rho_K(eps) of the 2D dispersion, with the
        central patch |kx|, |ky| < pi/sqrt(2) and its complement."""
        k = np.linspace(-np.pi, np.pi, self.n_k)
        kx, ky = np.meshgrid(k, k)
        epsk = -2 * self.t * (np.cos(kx) + np.cos(ky)) - 4 * self.tp * np.cos(kx) * np.cos(ky)
        central = (np.abs(kx) < np.pi / np.sqrt(2)) & (np.abs(ky) < np.pi / np.sqrt(2))

        eps_patch, rho_patch = [], []
        for mask in (central, np.invert(central)):
            counts, edges = np.histogram(np.extract(mask, epsk), bins=self.n_bins, density=True)
            centres = 0.5 * (edges[:-1] + edges[1:])
            weights = counts * (edges[1] - edges[0])
            eps_patch.append(float(np.sum(weights * centres)))
            rho_patch.append((centres, weights))

        # Each patch is exactly half the zone, so eps_even = -eps_odd exactly; the k-grid misses
        # this by ~1e-3. Restore it by moving the energy zero.
        self.eps_asymmetry = 0.5 * (eps_patch[0] + eps_patch[1])
        eps_patch = [e - self.eps_asymmetry for e in eps_patch]

        # The two densities of states have different widths, which breaks particle-hole symmetry
        self.dos_variance_ratio = float(
            np.sum(rho_patch[0][1] * (rho_patch[0][0] - eps_patch[0] - self.eps_asymmetry)**2)
            / np.sum(rho_patch[1][1] * (rho_patch[1][0] - eps_patch[1] - self.eps_asymmetry)**2))
        return eps_patch, rho_patch

    def delta_iw(self, n_iw):
        """Hybridization Delta_K(iw), diagonal in the patch index, with eps_K removed (it is in h_loc0)."""
        delta = BlockGf(name_list=list(SPIN_NAMES),
                        block_list=[GfImFreq(beta=self.beta, n_points=n_iw, target_shape=(N_PATCH, N_PATCH))
                                    for _ in SPIN_NAMES])
        for _, d in delta:
            d.zero()
            for K in range(N_PATCH):
                if self.bath == 'discrete':
                    d[K, K] << self.V**2 * inverse(iOmega_n - self.eps_bath)
                else:
                    G_K = GfImFreq(beta=self.beta, n_points=n_iw, target_shape=(1, 1))
                    G_K.zero()
                    centres, weights = self.rho_patch[K]
                    for eps, w in zip(centres, weights):
                        G_K << G_K + w * inverse(iOmega_n - eps)
                    d[K, K] << (iOmega_n - self.eps_patch[K] - inverse(G_K)[0, 0])
        return delta

    def half_filling_is_exact(self):
        """mu = U/2 is exactly half filling only for the particle-hole symmetric discrete bath."""
        return self.bath == 'discrete'

    # -----------------------------------------------------------------------------------

    def h_int(self):
        return self.U * sum(self.n_site(i, 'up') * self.n_site(i, 'down') for i in range(N_PATCH))

    def h_loc0(self):
        return sum((self.eps_patch[K] - self.mu) * c_dag(s, K) * c(s, K)
                   for s in SPIN_NAMES for K in range(N_PATCH))

    def Q(self, tau):
        return -np.cosh(self.omega_0 * (tau - self.beta / 2)) / (2 * self.omega_0 * np.sinh(self.omega_0 * self.beta / 2))

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
