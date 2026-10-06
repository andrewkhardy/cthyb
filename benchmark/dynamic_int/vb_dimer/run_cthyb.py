# CTHYB run of the two-patch model in model.py: Hubbard U and a retarded S.S interaction,
# both local on the rotated cluster sites, hence off-diagonal in the solver's patch basis.
#
#   mpirun -n <N> python run_cthyb.py --J_inter 0.5 --n_cycles 1000000
#
# Run check_rotation.py first: it checks the model symbolically in about a second.
#
# Observables: G_tau per patch, and O_tau = <S_tot^z(tau) S_tot^z(0)>. S_tot^z is the only
# spin operator commuting with h_loc here, which measure_O_tau requires.

import argparse
import os
import sys
from itertools import product

import numpy as np
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs.gfs import Fourier
from triqs_cthyb import Solver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import grids, selfenergy  # noqa: E402
import model as model_def
from model import N_PATCH, SPIN_NAMES

str_to_bool = lambda x: str(x).lower() in ['true', '1', 'yes']
parser = argparse.ArgumentParser(description='CTHYB: two-patch DCA with a retarded site-local S.S interaction.')
model_def.add_model_args(parser)
parser.add_argument('--n_cycles', type=int, default=1000000)
parser.add_argument('--n_warmup_cycles', type=int, default=50000)
parser.add_argument('--length_cycle', type=int, default=100)
parser.add_argument('--max_time', type=int, default=2700, help='Wall-clock cap in seconds, -1 to disable')
parser.add_argument('--n_l', type=int, default=50, help='Legendre coefficients for G_l')
parser.add_argument('--dyn_n_l', type=int, default=50, help='Legendre coefficients for the dynamical interaction')
parser.add_argument('--lang_firsov', type=str_to_bool, default=True,
                    help='False samples every vertex stochastically; the cross-check of the analytic route')
parser.add_argument('--measure_O_tau_min_ins', type=int, default=100)
parser.add_argument('--density_matrix', type=str_to_bool, default=False,
                    help='Measure the density matrix (and use_norm_as_weight)')
parser.add_argument('--random_seed', type=int, default=None, help='Base seed; rank r uses base + 928374 * r')
parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
args = parser.parse_args()
M = model_def.Model(args)

n_iw, n_tau = grids.N_IW, grids.N_TAU
n_tau_bosonic = n_tau
S = Solver(beta=M.beta, gf_struct=M.gf_struct, n_iw=n_iw, n_tau=n_tau, n_l=args.n_l,
           n_tau_bosonic=n_tau_bosonic, delta_interface=True)
delta_iw = M.delta_iw(n_iw)
for bl, delta in delta_iw:
    S.Delta_tau[bl] << Fourier(delta)

# Retarded spin-spin interaction on the cluster sites,
#   S_dyn = 1/2 sum_ij int int -J_ij Q(tau - tau') S_i(tau) . S_j(tau'),
#   S_i . S_j = S^z_i S^z_j + (S^+_i S^-_j + S^-_i S^+_j) / 2.
Q = M.Q(np.linspace(0, M.beta, n_tau_bosonic))
for i, j in product(range(N_PATCH), repeat=2):
    D = -M.J[i, j] * Q
    S.add_dyn_int(D, M.Sz[i], M.Sz[j])
    S.add_dyn_int(D / 2, M.Sp[i], M.Sm[j])
    S.add_dyn_int(D / 2, M.Sm[i], M.Sp[j])

# The static part of a spin-spin interaction is particle-hole even, so it does not move mu.
if not M.half_filling_is_exact() and mpi.is_master_node():
    print(f"NOTE: --bath {M.bath} is not particle-hole symmetric (patch DOS variance ratio "
          f"{M.dos_variance_ratio:.3f}), so mu = {M.mu} is only roughly half filling; "
          f"use calibrate_mu.py for a target density.")

S.solve(h_int=M.h_int(),
        h_loc0=M.h_loc0(),
        n_cycles=args.n_cycles,
        n_warmup_cycles=args.n_warmup_cycles,
        length_cycle=args.length_cycle,
        max_time=args.max_time,
        lang_firsov=args.lang_firsov,
        dyn_n_l=args.dyn_n_l,
        measure_G_l=True,
        measure_pert_order=True,
        measure_D0_corr=True,
        measure_O_tau=(M.Sz_total, M.Sz_total),
        measure_O_tau_min_ins=args.measure_O_tau_min_ins,
        measure_density_matrix=args.density_matrix,
        use_norm_as_weight=args.density_matrix,
        **({} if args.random_seed is None else dict(random_seed=args.random_seed + 928374 * mpi.rank)))

if mpi.is_master_node():
    fillings = {bl: [-g.data[-1, i, i].real for i in range(N_PATCH)] for bl, g in S.G_tau}
    total_filling = sum(sum(v) for v in fillings.values())
    print(f"average sign {S.average_sign}, average order {S.average_order}")
    print(f"patch fillings {dict((k, np.round(v, 5).tolist()) for k, v in fillings.items())}, "
          f"total <N> = {total_filling:.5f} (2.0 is half filling)")

    # Sigma from the input mu and Delta, not the solver's h_loc: solve() shifts h_loc by the
    # Lang-Firsov static part, so reading it back would differ between lang_firsov True and False.
    eps_patch = np.diag(M.eps_patch)
    sigma_l = selfenergy.sigma_from_G_l(S.G_l, n_iw, M.mu, delta_iw, eps_patch)
    sigma_tau = selfenergy.sigma_from_G_tau(S.G_tau, n_iw, M.mu, delta_iw, eps_patch)

    w_n = selfenergy.matsubara_frequencies(sigma_l[M.gf_struct[0][0]].mesh)
    sigma_orb = np.array([selfenergy.positive_frequency_part(sigma_l[s], (K, K))[1] for s, K in M.labels])
    sigma_orb_alt = np.array([selfenergy.positive_frequency_part(sigma_tau[s], (K, K))[1] for s, K in M.labels])
    density = selfenergy.density_from_G_iw(selfenergy.G_iw_from_G_l(S.G_l, n_iw))
    print("  " + selfenergy.diagnose(sigma_orb[0], w_n)["text"])
    print(f"  <n> = {np.round(density, 5)}")

    os.makedirs(args.out_dir, exist_ok=True)
    seed_tag = '' if args.random_seed is None else f"_seed-{args.random_seed}"
    filename = os.path.join(args.out_dir,
                            f"cthyb_{M.tag()}_lf-{args.lang_firsov}_nl-{args.n_l}_nc-{args.n_cycles}{seed_tag}.h5")
    with HDFArchive(filename, 'w') as A:
        # Same keys as run_ed.py
        A['tau'] = np.array([float(t) for t in S.G_tau[M.gf_struct[0][0]].mesh])
        A['G'] = np.array([S.G_tau[s].data[:, K, K].real for s, K in M.labels])
        A['w_n'] = w_n
        A['Sigma'] = sigma_orb
        A['Sigma_alt'] = sigma_orb_alt
        A['density'] = density
        A['labels'] = [f"{s},{K}" for s, K in M.labels]
        A['tau_corr'] = np.array([float(t) for t in S.O_tau.mesh])
        A['corr'] = np.asarray(S.O_tau.data).real.flatten()  # <S_tot^z(tau) S_tot^z(0)>
        A['G_tau'] = S.G_tau
        A['G_l'] = S.G_l
        A['O_tau'] = S.O_tau
        A['Q_conserved_tau'] = S.Q_conserved_tau
        A['conserved_operators'] = [str(op) for op in S.conserved_density_operators]
        A['average_sign'] = S.average_sign
        A['average_order'] = S.average_order
        A['fillings'] = np.array([fillings[s] for s in SPIN_NAMES])
        A['total_filling'] = total_filling
        if S.perturbation_order_dyn is not None:
            A['perturbation_order_dyn'] = S.perturbation_order_dyn
        A['perturbation_order'] = S.perturbation_order
        A['params'] = M.params()
        A['mu'] = M.mu
        A['lang_firsov'] = args.lang_firsov
        A['n_cycles'] = args.n_cycles
        A['n_l'] = args.n_l
        A['random_seed'] = -1 if args.random_seed is None else args.random_seed
    print(f"Saved {filename}")
