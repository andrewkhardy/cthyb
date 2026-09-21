# CTHYB run of the model in model.py (same Hamiltonian as run_ed.py): the bath enters
# as Delta_a(iw) = V^2 / (iw - eps_bath), the phonon as the retarded coupling
# D_ab(tau) = g_a g_b Q(tau) on every ordered pair of spin-orbitals, diagonal included.
# Routing is decided by the solver: equal g for both orbitals is exactly a coupling to
# N_up and N_down and goes to Lang-Firsov; unequal g currently goes fully stochastic.
#
#   mpirun -np <N> python run_cthyb.py --g 0.5 0.5 --n_cycles 1000000
#
# Compare with plot_kanamori_phonon.py.

import argparse
import os
import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import Fourier
from triqs.operators import n
from h5 import HDFArchive
from triqs_cthyb import Solver

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import selfenergy  # noqa: E402
import model as model_def

str_to_bool = lambda x: str(x).lower() in ['true', '1', 'yes']
parser = argparse.ArgumentParser(description='CTHYB: Kanamori impurity + bath + phonon (ED-comparable).')
model_def.add_model_args(parser)
parser.add_argument('--n_cycles', type=int, default=1000000)
parser.add_argument('--n_warmup_cycles', type=int, default=20000)
parser.add_argument('--length_cycle', type=int, default=100)
parser.add_argument('--max_time', type=int, default=2700,
                    help='Hard wall-clock cap in seconds for the MC, -1 to disable')
parser.add_argument('--n_l', type=int, default=50, help='Legendre coefficients for G_l (Sigma route)')
parser.add_argument('--dyn_n_l', type=int, default=50, help='Legendre coefficients for the dynamical interaction and Q')
parser.add_argument('--lang_firsov', type=str_to_bool, default=True, help='False forces every vertex stochastic')
parser.add_argument('--density_matrix', type=str_to_bool, default=True,
                    help='Measure the density matrix, so the equal-time <O_i O_j> is added back to Q_conserved_tau')
parser.add_argument('--random_seed', type=int, default=None,
                    help='Base seed; rank r uses base + 928374 * r. The solver default is fixed, so a plain '
                         'rerun is bit-identical: pass different values to get independent samples, which is '
                         'the only way to put an error bar on the l = 0 channel (diagnose_residual.py).')
parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
args = parser.parse_args()
M = model_def.Model(args)

n_iw, n_tau, n_tau_bosonic = 1025, 10001, 2001
S = Solver(beta=M.beta, gf_struct=M.gf_struct, n_iw=n_iw, n_tau=n_tau, n_l=args.n_l,
           n_tau_bosonic=n_tau_bosonic, delta_interface=True)
for bl, delta in M.delta_iw(n_iw):
    S.Delta_tau[bl] << Fourier(delta)

Q = M.Q(np.linspace(0, M.beta, n_tau_bosonic))
for a, (s1, o1) in enumerate(M.labels):
    for b, (s2, o2) in enumerate(M.labels):
        S.D0_tau[s1, s2].data[:, o1, o2] = M.g[a] * M.g[b] * Q

S.solve(h_int=M.h_int(),
        h_loc0=-sum(M.mu[a] * n(*M.labels[a]) for a in range(len(M.labels))),
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
        **({} if args.random_seed is None else dict(random_seed=args.random_seed + 928374 * mpi.rank)))

if mpi.is_master_node():
    # Conserved combinations as coefficient vectors in model.labels order, for the ED contraction
    conserved_vectors = np.zeros((len(S.conserved_density_operators), len(M.labels)))
    for i, op in enumerate(S.conserved_density_operators):
        for term, coeff in op:
            (_, (bl, idx)), _ = term
            conserved_vectors[i, M.labels.index((bl, idx))] = np.real(coeff)

    # Orbital-resolved correlators from the stochastic (residual) vertices: the coupling-derivative
    # estimator, <n_a(tau) n_b(0)> per vertex type. dyn_vertex_pairs gives each type's (a, b) as
    # indices into model.labels, or -1 for a type that is not a density-density vertex.
    dyn_vertex_pairs, dyn_vertex_corr = [], []
    if S.dyn_vertex_corr_tau is not None:
        def density_orbital(op):
            (term, _), = list(op)
            (_, indices_dag), (_, indices) = term[0], term[1]
            return M.labels.index(tuple(indices_dag)) if list(indices_dag) == list(indices) else -1

        for (op1, op2), g in zip(S.dyn_vertex_operators, S.dyn_vertex_corr_tau):
            dyn_vertex_pairs.append([density_orbital(op1), density_orbital(op2)])
            dyn_vertex_corr.append(g.data.real)

    # The equal-time constant solver.py adds to Q_conserved_l[0], and the densities behind it. Both
    # come from the density matrix, not from the kink estimator, and the l = 0 coefficient is where
    # almost all of the CTHYB - ED residual lives: save them so that channel can be checked on its
    # own against ED (sum_ab v_i[a] v_j[b] chi_ab(0) and <n_a> = 1/2).
    if args.density_matrix:
        from triqs.atom_diag import trace_rho_op
        ops = S.conserved_density_operators
        A_equal_time = np.array([[trace_rho_op(S.density_matrix, Oi * Oj, S.h_loc_diagonalization).real
                                  for Oj in ops] for Oi in ops])
        A_occupations = np.array([np.real(S.orbital_occupations[bl][o, o]) for bl, o in M.labels])

    # --- Self-energy, in the same convention as every other benchmark here.
    # G0 is built from the *input* mu and Delta (common/selfenergy), never read back from
    # the solver: solve() mutates h_loc by the Lang-Firsov K'(0) shift, so a mu recovered
    # afterwards is route-dependent and would differ between lang_firsov True and False.
    # Preferred route is the Legendre G_l; the G(tau) Dyson inversion is kept as the
    # cross-check, since the two bracket the systematic (the Legendre one stays causal much
    # further up the Matsubara axis).
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

    os.makedirs(args.out_dir, exist_ok=True)
    seed_tag = '' if args.random_seed is None else f"_seed-{args.random_seed}"
    filename = os.path.join(args.out_dir, f"cthyb_{M.tag()}_lf-{args.lang_firsov}_nc-{args.n_cycles}{seed_tag}.h5")
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
        if len(dyn_vertex_pairs) > 0:
            A['dyn_vertex_pairs'] = np.array(dyn_vertex_pairs)
            A['dyn_vertex_corr'] = np.array(dyn_vertex_corr)
            A['dyn_vertex_tau'] = np.linspace(0, M.beta, n_tau_bosonic)
        if S.perturbation_order_dyn is not None:
            A['perturbation_order_dyn'] = S.perturbation_order_dyn
        A['params'] = M.params()
        A['mu'] = M.mu
        A['lang_firsov'] = args.lang_firsov
        A['n_cycles'] = args.n_cycles
        A['random_seed'] = -1 if args.random_seed is None else args.random_seed
        if args.density_matrix:
            A['equal_time_conserved'] = A_equal_time
            A['orbital_occupations'] = A_occupations
    print(f"Saved {filename}, average sign {S.average_sign}")
