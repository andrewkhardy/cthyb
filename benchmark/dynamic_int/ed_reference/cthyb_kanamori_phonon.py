# CTHYB run of the model in model.py (same Hamiltonian as ed_kanamori_phonon.py): the bath enters
# as Delta_a(iw) = V^2 / (iw - eps_bath), the phonon as the retarded coupling
# D_ab(tau) = g_a g_b Q(tau) on every ordered pair of spin-orbitals, diagonal included.
# Routing is decided by the solver: equal g for both orbitals is exactly a coupling to
# N_up and N_down and goes to Lang-Firsov; unequal g currently goes fully stochastic.
#
#   mpirun -np <N> python cthyb_kanamori_phonon.py --g 0.5 0.5 --n_cycles 1000000
#
# Compare with plot_ed_vs_cthyb.py.

import argparse
import os
import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import Fourier
from triqs.operators import n
from h5 import HDFArchive
from triqs_cthyb import Solver
import model as model_def

str_to_bool = lambda x: str(x).lower() in ['true', '1', 'yes']
parser = argparse.ArgumentParser(description='CTHYB: Kanamori impurity + bath + phonon (ED-comparable).')
model_def.add_model_args(parser)
parser.add_argument('--n_cycles', type=int, default=1000000)
parser.add_argument('--n_warmup_cycles', type=int, default=20000)
parser.add_argument('--length_cycle', type=int, default=100)
parser.add_argument('--dyn_n_l', type=int, default=50, help='Legendre coefficients for the dynamical interaction and Q')
parser.add_argument('--lang_firsov', type=str_to_bool, default=True, help='False forces every vertex stochastic')
parser.add_argument('--density_matrix', type=str_to_bool, default=True,
                    help='Measure the density matrix, so the equal-time <O_i O_j> is added back to Q_conserved_tau')
parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
args = parser.parse_args()
M = model_def.Model(args)

n_iw, n_tau, n_tau_bosonic = 1025, 10001, 2001
S = Solver(beta=M.beta, gf_struct=M.gf_struct, n_iw=n_iw, n_tau=n_tau, n_tau_bosonic=n_tau_bosonic, delta_interface=True)
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
        lang_firsov=args.lang_firsov,
        dyn_n_l=args.dyn_n_l,
        measure_D0_corr=True,
        measure_pert_order=True,
        measure_density_matrix=args.density_matrix,
        use_norm_as_weight=args.density_matrix)

if mpi.is_master_node():
    # Conserved combinations as coefficient vectors in model.labels order, for the ED contraction
    conserved_vectors = np.zeros((len(S.conserved_density_operators), len(M.labels)))
    for i, op in enumerate(S.conserved_density_operators):
        for term, coeff in op:
            (_, (bl, idx)), _ = term
            conserved_vectors[i, M.labels.index((bl, idx))] = np.real(coeff)

    os.makedirs(args.out_dir, exist_ok=True)
    filename = os.path.join(args.out_dir, f"cthyb_{M.tag()}_lf-{args.lang_firsov}_nc-{args.n_cycles}.h5")
    with HDFArchive(filename, 'w') as A:
        A['G_tau'] = S.G_tau
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
    print(f"Saved {filename}, average sign {S.average_sign}")
