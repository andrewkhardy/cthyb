# CTHYB run of the model in model.py (same Hamiltonian as run_ed.py): the bath enters as
# Delta_a(iw) = V^2 / (iw - eps_bath), the phonon as S_dyn = 1/2 int int Q(tau - tau') X(tau) X(tau')
# with X = sum_a g_a n_a. Equal g is a coupling to N_up and N_down and goes entirely to Lang-Firsov;
# unequal g leaves a residual that is sampled stochastically.
#
#   mpirun -np <N> python run_cthyb.py --g 0.5 0.5 --n_cycles 1000000

import argparse
import os
import time
import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import Fourier
from triqs.operators import n
from h5 import HDFArchive
from triqs_cthyb import Solver

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import grids, selfenergy, str2bool  # noqa: E402
import model as model_def

parser = argparse.ArgumentParser(description='CTHYB: Kanamori impurity + bath + phonon (ED-comparable).')
model_def.add_model_args(parser)
parser.add_argument('--n_cycles', type=int, default=1000000)
parser.add_argument('--n_warmup_cycles', type=int, default=20000)
parser.add_argument('--length_cycle', type=int, default=100)
parser.add_argument('--max_time', type=int, default=2700, help='MC wall-clock cap in seconds, -1 for none')
parser.add_argument('--n_l', type=int, default=50, help='Legendre coefficients for G_l (Sigma route)')
parser.add_argument('--dyn_n_l', type=int, default=50, help='Legendre coefficients for the dynamical interaction and Q')
parser.add_argument('--lang_firsov', type=str2bool, default=True, help='False forces every vertex stochastic')
parser.add_argument('--density_matrix', type=str2bool, default=True,
                    help='Measure the density matrix, so the equal-time <O_i O_j> is added back to Q_conserved_tau')
parser.add_argument('--measure_O_tau', type=int, nargs=2, default=None, metavar=('A', 'B'),
                    help='Measure O_tau = <n_B(tau) n_A(0)>, A and B indices into model.labels (ED: chi[B, A])')
parser.add_argument('--measure_nn_tau', type=str2bool, default=False,
                    help='Measure nn_tau = <n_a(tau) n_b(0)> for every pair, saved in model.labels order (ED: chi[a, b])')
parser.add_argument('--random_seed', type=int, default=None,
                    help='Base seed, rank r uses base + 928374 * r (default: the solver\'s fixed seed)')
parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
args = parser.parse_args()
M = model_def.Model(args)

# One tau grid for G, the kernel and Q, shared with run_ed.py (common/grids.py).
n_iw, n_tau = grids.N_IW, grids.N_TAU
n_tau_bosonic = n_tau
S = Solver(beta=M.beta, gf_struct=M.gf_struct, n_iw=n_iw, n_tau=n_tau, n_l=args.n_l,
           n_tau_bosonic=n_tau_bosonic, delta_interface=True)
for bl, delta in M.delta_iw(n_iw):
    S.Delta_tau[bl] << Fourier(delta)

# Phonon: S_dyn = 1/2 int int Q(tau - tau') X(tau) X(tau'), with X = sum_a g_a n_a
Q = M.Q(np.linspace(0, M.beta, n_tau_bosonic))
S.add_dyn_int(Q, M.X, M.X)

O_tau_args = {}
if args.measure_O_tau is not None:
    A_idx, B_idx = args.measure_O_tau
    O_tau_args = dict(measure_O_tau=(n(*M.labels[A_idx]), n(*M.labels[B_idx])))

solve_start = time.perf_counter()
S.solve(h_int=M.h_int(),
        h_loc0=M.h_loc0(),
        n_cycles=args.n_cycles,
        n_warmup_cycles=args.n_warmup_cycles,
        length_cycle=args.length_cycle,
        max_time=args.max_time,
        lang_firsov=args.lang_firsov,
        dyn_n_l=args.dyn_n_l,
        measure_G_tau=True,
        measure_G_l=True,
        measure_D0_corr=True,
        measure_pert_order=True,
        measure_density_matrix=args.density_matrix,
        use_norm_as_weight=args.density_matrix,
        measure_nn_tau=args.measure_nn_tau,
        **O_tau_args,
        **({} if args.random_seed is None else dict(random_seed=args.random_seed + 928374 * mpi.rank)))
solve_seconds = time.perf_counter() - solve_start

if mpi.is_master_node():
    # Conserved combinations as coefficient vectors in model.labels order, for the ED contraction
    conserved_vectors = np.zeros((len(S.conserved_density_operators), len(M.labels)))
    for i, op in enumerate(S.conserved_density_operators):
        for term, coeff in op:
            (_, (bl, idx)), _ = term
            conserved_vectors[i, M.labels.index((bl, idx))] = np.real(coeff)

    # The equal-time constant solver.py adds to Q_conserved_l[0], and the occupations, both from
    # the density matrix: they let the l = 0 channel be checked against ED on its own.
    if args.density_matrix:
        from triqs.atom_diag import trace_rho_op
        ops = S.conserved_density_operators
        A_equal_time = np.array([[trace_rho_op(S.density_matrix, Oi * Oj, S.h_loc_diagonalization).real
                                  for Oj in ops] for Oi in ops])
        A_occupations = np.array([np.real(S.orbital_occupations[bl][o, o]) for bl, o in M.labels])

    # Sigma from the *input* mu and Delta (common/selfenergy): G_l preferred, G(tau) the cross-check.
    delta_block = M.delta_iw(n_iw)
    mu_per_block = {bl: np.array([M.mu[a] for a, (s, o) in enumerate(M.labels) if s == bl])
                    for bl, _ in M.gf_struct}
    sigma_l = selfenergy.sigma_from_G_l(S.G_l, n_iw, mu_per_block, delta_block)
    sigma_tau = selfenergy.sigma_from_G_tau(S.G_tau, n_iw, mu_per_block, delta_block)

    # One curve per spin-orbital, in M.labels order, to line up with the ED output.
    w_n = selfenergy.matsubara_frequencies(sigma_l[M.gf_struct[0][0]].mesh)
    sigma_orb = np.array([selfenergy.positive_frequency_part(sigma_l[s], (o, o))[1]
                          for s, o in M.labels])
    sigma_orb_alt = np.array([selfenergy.positive_frequency_part(sigma_tau[s], (o, o))[1]
                              for s, o in M.labels])
    density = selfenergy.density_from_G_iw(selfenergy.G_iw_from_G_l(S.G_l, n_iw))
    print("  " + selfenergy.diagnose(sigma_orb[0], w_n)["text"])
    print(f"  <n> = {np.round(density, 5)}")

    # Equilibrium check, to read before Sigma: the Dyson inversion amplifies an unequilibrated
    # G by ~1/|G(i w_0)|^2, while G(tau) itself still looks right.
    n_orb = len(M.labels) // 2
    spin_asym = np.abs(density[:n_orb] - density[n_orb:]).max()
    text = (f"auto-correlation time {S.auto_corr_time:.0f} cycles"
            + ("" if S.auto_corr_time_converged else " (lower bound)")
            + f" vs warmup {args.n_warmup_cycles}, n_cycles {args.n_cycles}; "
            + f"max |n_up - n_down| {spin_asym:.4f}")
    suspect = S.auto_corr_time > args.n_warmup_cycles or spin_asym > 5e-3
    if args.density_matrix:
        estimator_gap = np.abs(A_occupations - density).max()
        text += f"; max |n_rho - n_G| {estimator_gap:.4f}"
        suspect = suspect or estimator_gap > 5e-3
    print(("  WARNING, likely not equilibrated -- " if suspect else "  equilibrium: ") + text)

    os.makedirs(args.out_dir, exist_ok=True)
    seed_tag = '' if args.random_seed is None else f"_seed-{args.random_seed}"
    O_tau_tag = '' if args.measure_O_tau is None else f"_Otau-{args.measure_O_tau[0]}-{args.measure_O_tau[1]}"
    nn_tau_tag = '_nntau' if args.measure_nn_tau else ''
    filename = os.path.join(args.out_dir, f"cthyb_{M.tag()}_lf-{args.lang_firsov}_nc-{args.n_cycles}{O_tau_tag}{nn_tau_tag}{seed_tag}.h5")
    with HDFArchive(filename, 'w') as A:
        A['G_tau'] = S.G_tau
        A['G_l'] = S.G_l
        A['w_n'] = w_n
        A['Sigma'] = sigma_orb
        A['Sigma_alt'] = sigma_orb_alt
        A['density'] = density
        A['Q_conserved_tau'] = S.Q_conserved_tau
        A['conserved_vectors'] = conserved_vectors
        A['conserved_operators'] = [str(op) for op in S.conserved_density_operators]
        A['equal_time_added'] = args.density_matrix
        A['average_sign'] = S.average_sign
        A['average_order'] = S.average_order
        if S.perturbation_order_dyn is not None:
            A['perturbation_order_dyn'] = S.perturbation_order_dyn
        A['params'] = M.params()
        A['mu'] = M.mu
        A['lang_firsov'] = args.lang_firsov
        A['n_cycles'] = args.n_cycles
        A['n_warmup_cycles'] = args.n_warmup_cycles
        A['length_cycle'] = args.length_cycle
        A['auto_corr_time'] = S.auto_corr_time
        A['auto_corr_time_converged'] = S.auto_corr_time_converged
        A['random_seed'] = -1 if args.random_seed is None else args.random_seed
        A['solve_seconds'] = solve_seconds
        if args.measure_O_tau is not None:
            A['O_tau'] = S.O_tau
            A['O_tau_pair'] = np.array(args.measure_O_tau)
        if args.measure_nn_tau:
            A['nn_tau'] = np.array([[S.nn_tau[s1, s2].data[:, o1, o2].real for s2, o2 in M.labels]
                                    for s1, o1 in M.labels])
            A['nn_tau_tau'] = np.array([float(t) for t in S.nn_tau[M.labels[0][0], M.labels[0][0]].mesh])
        if args.density_matrix:
            A['equal_time_conserved'] = A_equal_time
            A['orbital_occupations'] = A_occupations
    print(f"Saved {filename}, average sign {S.average_sign}")
