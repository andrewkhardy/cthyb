# Shared model definition for the exact-diagonalization reference (ed_kanamori_phonon.py) and
# the matching CTHYB run (cthyb_kanamori_phonon.py), so both solve exactly the same problem:
#
#   H = H_kanamori - sum_a mu_a n_a                               two orbitals x two spins, with
#                                                                 spin-flip and pair-hopping
#     + sum_a [eps_bath b_a^dag b_a + V (c_a^dag b_a + h.c.)]     one bath site per spin-orbital
#     + omega_0 d^dag d + (sum_a g_a n_a) (d + d^dag) / sqrt(2 omega_0)
#
# Integrating out the bath gives Delta_a(iw) = V^2 / (iw - eps_bath); integrating out the phonon
# gives the retarded density-density coupling D_ab(tau) = g_a g_b Q(tau), with
# Q(tau) = -cosh(omega_0 (tau - beta/2)) / (2 omega_0 sinh(omega_0 beta/2))
# (doc/notes/dynamical_interactions.tex, Eqs. holstein and Qtau). That D_ab and Delta_a are the
# CTHYB input; the ED solves H directly. Unequal g for the two orbitals gives the non-uniform
# "U_tot + dU" case.

from itertools import product
import numpy as np
from triqs.gfs import BlockGf, GfImFreq, inverse, iOmega_n
from triqs.operators.util.hamiltonians import h_int_kanamori

SPIN_NAMES = ('up', 'down')
N_ORB = 2


def add_model_args(parser):
    parser.add_argument('--beta', type=float, default=5.0, help='Inverse temperature')
    parser.add_argument('--U', type=float, default=2.0, help='Intra-orbital Hubbard U')
    parser.add_argument('--J', type=float, default=0.3,
                        help="Hund's coupling: U' = U - 2J, same-spin U - 3J, spin-flip and pair-hopping included")
    parser.add_argument('--V', type=float, default=0.7, help='Hybridization to the bath site of each spin-orbital')
    parser.add_argument('--eps_bath', type=float, default=0.0, help='Bath-site energy')
    parser.add_argument('--omega_0', type=float, default=1.0, help='Phonon frequency')
    parser.add_argument('--g', type=float, nargs=2, default=[0.5, 0.5], metavar=('G_ORB0', 'G_ORB1'),
                        help='Phonon coupling of orbital 0 and orbital 1 (same for both spins)')
    parser.add_argument('--mu', type=float, default=None,
                        help='Chemical potential of every spin-orbital; default: half filling, phonon static shift included')


class Model:

    def __init__(self, args):
        self.beta, self.U, self.J = args.beta, args.U, args.J
        self.V, self.eps_bath, self.omega_0 = args.V, args.eps_bath, args.omega_0
        self.g_orb = list(args.g)
        self.gf_struct = [(s, N_ORB) for s in SPIN_NAMES]
        # Spin-orbitals in CTHYB's linear-index order for this gf_struct
        self.labels = list(product(SPIN_NAMES, range(N_ORB)))
        self.g = np.array([self.g_orb[o] for s, o in self.labels])

        # Static density-density interaction, Kanamori plus the phonon's instantaneous part
        # -(sum_a g_a n_a)^2 / (2 omega_0^2): -g_a g_b / omega_0^2 per pair, -g_a^2 / (2 omega_0^2) n_a
        n_so = len(self.labels)
        W = np.zeros((n_so, n_so))
        for a, (s1, o1) in enumerate(self.labels):
            for b, (s2, o2) in enumerate(self.labels):
                if a == b: continue
                kanamori = self.U if o1 == o2 else (self.U - 3 * self.J if s1 == s2 else self.U - 2 * self.J)
                W[a, b] = kanamori - self.g[a] * self.g[b] / self.omega_0**2
        self.mu_is_half_filling = args.mu is None
        if self.mu_is_half_filling:
            # Particle-hole symmetry: mu_a + g_a^2 / (2 omega_0^2) = (1/2) sum_{b != a} W_ab
            self.mu = 0.5 * W.sum(axis=1) - self.g**2 / (2 * self.omega_0**2)
        else:
            self.mu = np.full(n_so, args.mu)

    def h_int(self):
        U_same_spin = np.array([[0 if o1 == o2 else self.U - 3 * self.J for o2 in range(N_ORB)] for o1 in range(N_ORB)])
        U_opposite_spin = np.array([[self.U if o1 == o2 else self.U - 2 * self.J for o2 in range(N_ORB)] for o1 in range(N_ORB)])
        return h_int_kanamori(SPIN_NAMES, N_ORB, U_same_spin, U_opposite_spin, self.J, off_diag=True)

    def Q(self, tau):
        return -np.cosh(self.omega_0 * (tau - self.beta / 2)) / (2 * self.omega_0 * np.sinh(self.omega_0 * self.beta / 2))

    def delta_iw(self, n_iw):
        delta = BlockGf(name_list=list(SPIN_NAMES),
                        block_list=[GfImFreq(beta=self.beta, n_points=n_iw, target_shape=(N_ORB, N_ORB)) for _ in SPIN_NAMES])
        for bl, d in delta:
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
