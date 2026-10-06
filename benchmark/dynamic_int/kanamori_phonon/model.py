# Two-orbital Kanamori impurity with one bath site per spin-orbital and a Holstein phonon.
# Shared by run_ed.py (which solves H directly) and run_cthyb.py:
#
#   H = H_kanamori - sum_a mu_a n_a
#     + sum_a [eps_bath b_a^dag b_a + V (c_a^dag b_a + h.c.)]
#     + omega_0 d^dag d + X (d + d^dag) / sqrt(2 omega_0),       X = sum_a g_a n_a
#
# Integrating out the bath gives Delta_a(iw) = V^2 / (iw - eps_bath). Integrating out the
# phonon gives the retarded interaction
#
#   S_dyn = 1/2 int int Q(tau - tau') X(tau) X(tau'),
#   Q(tau) = -cosh(omega_0 (tau - beta/2)) / (2 omega_0 sinh(omega_0 beta/2)).
#
# Unequal g on the two orbitals breaks the orbital symmetry of the phonon coupling.

from itertools import product
import numpy as np
from triqs.gfs import BlockGf, GfImFreq, inverse, iOmega_n
from triqs.operators import n
from triqs.operators.util.hamiltonians import h_int_kanamori

SPIN_NAMES = ('up', 'down')
N_ORB = 2


def add_model_args(parser):
    parser.add_argument('--beta', type=float, default=5.0, help='Inverse temperature')
    parser.add_argument('--U', type=float, default=2.0, help='Intra-orbital Hubbard U')
    parser.add_argument('--J', type=float, default=0.3, help="Hund's coupling (U' = U - 2J, with spin-flip and pair-hopping)")
    parser.add_argument('--V', type=float, default=0.7, help='Hybridization to the bath site of each spin-orbital')
    parser.add_argument('--eps_bath', type=float, default=0.0, help='Bath-site energy')
    parser.add_argument('--omega_0', type=float, default=1.0, help='Phonon frequency')
    parser.add_argument('--g', type=float, nargs=2, default=[0.5, 0.5], metavar=('G_ORB0', 'G_ORB1'),
                        help='Phonon coupling of orbital 0 and orbital 1')
    parser.add_argument('--mu', type=float, default=None,
                        help='Chemical potential of every spin-orbital; default half filling')


class Model:

    def __init__(self, args):
        self.beta, self.U, self.J = args.beta, args.U, args.J
        self.V, self.eps_bath, self.omega_0 = args.V, args.eps_bath, args.omega_0
        self.g_orb = list(args.g)
        self.gf_struct = [(s, N_ORB) for s in SPIN_NAMES]
        self.labels = list(product(SPIN_NAMES, range(N_ORB)))
        self.g = np.array([self.g_orb[o] for s, o in self.labels])

        # The operator the phonon couples to
        self.X = sum(self.g[a] * n(s, o) for a, (s, o) in enumerate(self.labels))

        # Half filling: the static density-density matrix is Kanamori plus the phonon's
        # instantaneous part -X^2 / (2 omega_0^2), and particle-hole symmetry fixes
        # mu_a + g_a^2 / (2 omega_0^2) = (1/2) sum_{b != a} W_ab.
        n_so = len(self.labels)
        W = np.zeros((n_so, n_so))
        for a, (s1, o1) in enumerate(self.labels):
            for b, (s2, o2) in enumerate(self.labels):
                if a == b:
                    continue
                kanamori = self.U if o1 == o2 else (self.U - 3 * self.J if s1 == s2 else self.U - 2 * self.J)
                W[a, b] = kanamori - self.g[a] * self.g[b] / self.omega_0**2
        self.mu_is_half_filling = args.mu is None
        if self.mu_is_half_filling:
            self.mu = 0.5 * W.sum(axis=1) - self.g**2 / (2 * self.omega_0**2)
        else:
            self.mu = np.full(n_so, args.mu)

    def h_int(self):
        U_same_spin = np.array([[0 if o1 == o2 else self.U - 3 * self.J for o2 in range(N_ORB)] for o1 in range(N_ORB)])
        U_opposite_spin = np.array([[self.U if o1 == o2 else self.U - 2 * self.J for o2 in range(N_ORB)] for o1 in range(N_ORB)])
        return h_int_kanamori(SPIN_NAMES, N_ORB, U_same_spin, U_opposite_spin, self.J, off_diag=True)

    def h_loc0(self):
        return -sum(self.mu[a] * n(s, o) for a, (s, o) in enumerate(self.labels))

    def Q(self, tau):
        return -np.cosh(self.omega_0 * (tau - self.beta / 2)) / (2 * self.omega_0 * np.sinh(self.omega_0 * self.beta / 2))

    def delta_iw(self, n_iw):
        delta = BlockGf(name_list=list(SPIN_NAMES),
                        block_list=[GfImFreq(beta=self.beta, n_points=n_iw, target_shape=(N_ORB, N_ORB)) for _ in SPIN_NAMES])
        for _, d in delta:
            d.zero()
            for o in range(N_ORB):
                d[o, o] << self.V**2 * inverse(iOmega_n - self.eps_bath)
        return delta

    def params(self):
        return dict(beta=self.beta, U=self.U, J=self.J, V=self.V, eps_bath=self.eps_bath, omega_0=self.omega_0,
                    g_orb0=self.g_orb[0], g_orb1=self.g_orb[1])

    def tag(self):
        mu = 'half' if self.mu_is_half_filling else f'{self.mu[0]}'
        return (f'beta-{self.beta}_U-{self.U}_J-{self.J}_V-{self.V}_eb-{self.eps_bath}_w0-{self.omega_0}'
                f'_g-{self.g_orb[0]}-{self.g_orb[1]}_mu-{mu}')
