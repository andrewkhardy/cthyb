# CTHYB run of the two-patch DCA model in model.py: two patch orbitals, a static Hubbard U
# local on the two *rotated* cluster sites, and a retarded full S.S written in the site basis.
#
#   mpirun -n <N> python run_cthyb.py --J_inter 0.5 --n_cycles 1000000
#
# The dynamical vertices are off-diagonal in the working basis (c^dag_K c_K' with K != K'), so
# this exercises the general stochastic expansion; only the part of the density coupling in
# span{N_up, N_down} can go to Lang-Firsov, and --lang_firsov False samples all of it.
#
# Saved: G, Sigma (from G_l) and Sigma_alt (from G(tau)) per patch orbital; corr = O_tau =
# <S^z_tot(tau) S^z_tot(0)> (measure_O_tau needs [O, h_loc] = 0, which the site and patch spins
# fail); and vertex_corr, the coupling-derivative estimator <op1(tau) op2(0)> of every stochastic
# vertex, from which the site-resolved <S_i(tau).S_j(0)> can be rebuilt (complete only with
# --lang_firsov False).

import argparse
import os
import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import Fourier
from h5 import HDFArchive
from triqs_cthyb import Solver

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import grids, selfenergy, str2bool  # noqa: E402
import model as model_def
from model import key_to_string, N_PATCH

parser = argparse.ArgumentParser(description='CTHYB: two-patch DCA with a real-space retarded S.S interaction.')
model_def.add_model_args(parser)
parser.add_argument('--n_cycles', type=int, default=1000000)
parser.add_argument('--n_warmup_cycles', type=int, default=50000)
parser.add_argument('--length_cycle', type=int, default=100)
parser.add_argument('--max_time', type=int, default=2700, help='MC wall-clock cap in seconds, -1 for none')
parser.add_argument('--n_l', type=int, default=50, help='Legendre coefficients for G_l (Sigma route)')
parser.add_argument('--dyn_n_l', type=int, default=50, help='Legendre coefficients for the dynamical interaction')
parser.add_argument('--lang_firsov', type=str2bool, default=True,
                    help='False samples every vertex stochastically, the cross-check of the analytic path')
parser.add_argument('--density_matrix', type=str2bool, default=False,
                    help='Measure the atomic density matrix (and use_norm_as_weight); nothing here uses it')
parser.add_argument('--random_seed', type=int, default=None,
                    help='Base seed, rank r uses base + 928374 * r (default: the solver\'s fixed seed)')
parser.add_argument('--dry_run', type=str2bool, default=False,
                    help='Print the registered vertex list and stop before the Monte Carlo')
parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
args = parser.parse_args()
M = model_def.Model(args)

# One tau grid for G, the kernel and O_tau, shared with run_ed.py (common/grids.py).
n_iw, n_tau = grids.N_IW, grids.N_TAU
n_tau_bosonic = n_tau
S = Solver(beta=M.beta, gf_struct=M.gf_struct, n_iw=n_iw, n_tau=n_tau, n_l=args.n_l,
           n_tau_bosonic=n_tau_bosonic, delta_interface=True)
# The *input* hybridization, which Sigma is built from below.
delta_iw = M.delta_iw(n_iw)
for bl, delta in delta_iw:
    S.Delta_tau[bl] << Fourier(delta)

registered = M.register_vertices(S, n_tau_bosonic, basis='site')
if mpi.is_master_node():
    print(f"Registered {len(registered)} dynamical vertices from the site-basis S.S "
          f"(J_intra = {M.J_intra}, J_inter = {M.J_inter}), patch levels {np.round(M.eps_patch, 5)}")

if args.dry_run:
    if mpi.is_master_node():
        for op1, op2, coeff in registered:
            print(f"  {coeff:+.5f}   ({op1}) (tau) * ({op2}) (0)")
    raise SystemExit(0)

# mu is never read back from the solver: what it routes analytically is route dependent, and the
# lf=True and lf=False runs must solve one Hamiltonian. The retarded S.S has no static charge part.
if not M.half_filling_is_exact() and mpi.is_master_node():
    print(f"NOTE: --bath {M.bath} breaks particle-hole symmetry (patch DOS variance ratio "
          f"{M.params()['dos_variance_ratio']:.3f}), so mu = {M.mu} is only approximately half "
          f"filling; calibrate --mu (calibrate_mu.py).")

S.solve(h_int=M.h_int(), h_loc0=M.h_loc0(),
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
        measure_density_matrix=args.density_matrix,
        use_norm_as_weight=args.density_matrix,
        **({} if args.random_seed is None else dict(random_seed=args.random_seed + 928374 * mpi.rank)))

if mpi.is_master_node():
    fillings = {bl: [-g.data[-1, i, i].real for i in range(N_PATCH)] for bl, g in S.G_tau}
    total_filling = sum(sum(v) for v in fillings.values())
    print(f"average sign {S.average_sign}, average order {S.average_order}")
    print(f"patch fillings {dict((k, np.round(v, 5).tolist()) for k, v in fillings.items())}, "
          f"total <N> = {total_filling:.5f} (2.0 is half filling)")

    # Sigma from the inputs, G0^-1 = iw + mu - eps_patch - Delta(iw) (delta_iw has no eps_patch):
    # G_l preferred, G(tau) the cross-check.
    eps_patch = np.diag(M.eps_patch)
    sigma_l = selfenergy.sigma_from_G_l(S.G_l, n_iw, M.mu, delta_iw, eps_patch)
    sigma_tau = selfenergy.sigma_from_G_tau(S.G_tau, n_iw, M.mu, delta_iw, eps_patch)

    # One curve per spin-orbital, in M.labels order, so it lines up with the ED output.
    w_n = selfenergy.matsubara_frequencies(sigma_l[M.gf_struct[0][0]].mesh)
    sigma_orb = np.array([selfenergy.positive_frequency_part(sigma_l[s], (K, K))[1]
                          for s, K in M.labels])
    sigma_orb_alt = np.array([selfenergy.positive_frequency_part(sigma_tau[s], (K, K))[1]
                              for s, K in M.labels])
    density = selfenergy.density_from_G_iw(selfenergy.G_iw_from_G_l(S.G_l, n_iw))
    print("  " + selfenergy.diagnose(sigma_orb[0], w_n)["text"])
    print(f"  <n> = {np.round(density, 5)}")

    # Vertex correlators labelled by their monomial pair, so they need no ordering convention.
    def operator_key(op):
        (mono, _), = list(op)
        return tuple((bool(d), tuple(i)) for d, i in mono)

    vertex_labels, vertex_corr = [], []
    if S.dyn_vertex_corr_tau is not None:
        for (op1, op2), g in zip(S.dyn_vertex_operators, S.dyn_vertex_corr_tau):
            vertex_labels.append([key_to_string(operator_key(op1)), key_to_string(operator_key(op2))])
            vertex_corr.append(g.data.real)
        print(f"Measured {len(vertex_corr)} stochastic vertex correlators "
              f"(of {len(registered)} registered)")

    os.makedirs(args.out_dir, exist_ok=True)
    seed_tag = '' if args.random_seed is None else f"_seed-{args.random_seed}"
    filename = os.path.join(args.out_dir,
                            f"cthyb_{M.tag()}_lf-{args.lang_firsov}_nl-{args.n_l}_nc-{args.n_cycles}{seed_tag}.h5")
    with HDFArchive(filename, 'w') as A:
        # Same key names as run_ed.py, so plot_vb_dimer.py reads both sides the same way.
        A['tau'] = np.array([float(t) for t in S.G_tau[M.gf_struct[0][0]].mesh])
        A['G'] = np.array([S.G_tau[s].data[:, K, K].real for s, K in M.labels])
        A['w_n'] = w_n
        A['Sigma'] = sigma_orb
        A['Sigma_alt'] = sigma_orb_alt
        A['density'] = density
        A['labels'] = [f"{s},{K}" for s, K in M.labels]
        # <S^z_tot(tau) S^z_tot(0)> = sum_ij of run_ed.py's chi_zz. Not from vertex_corr, which
        # divides by the coupling and is zeroed where it is negligible (most of tau at beta = 100).
        A['tau_corr'] = np.array([float(t) for t in S.O_tau.mesh])
        A['corr'] = np.asarray(S.O_tau.data).real.flatten()
        A['G_tau'] = S.G_tau
        A['G_l'] = S.G_l
        A['O_tau'] = S.O_tau
        A['Q_conserved_tau'] = S.Q_conserved_tau
        A['conserved_operators'] = [str(op) for op in S.conserved_density_operators]
        A['average_sign'] = S.average_sign
        A['average_order'] = S.average_order
        A['fillings'] = np.array([fillings[s] for s in model_def.SPIN_NAMES])
        A['total_filling'] = total_filling
        # The registered expansion, so the analysis can rebuild any site-basis correlator
        A['registered_labels'] = [[key_to_string(operator_key(o1)), key_to_string(operator_key(o2))]
                                  for o1, o2, _ in registered]
        A['registered_coeffs'] = np.array([coeff for _, _, coeff in registered])
        if len(vertex_corr) > 0:
            A['vertex_labels'] = vertex_labels
            A['vertex_corr'] = np.array(vertex_corr)
            A['vertex_tau'] = np.linspace(0, M.beta, n_tau_bosonic)
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
