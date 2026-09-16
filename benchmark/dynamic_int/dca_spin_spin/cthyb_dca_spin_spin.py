# CTHYB run of the two-patch DCA model in dca_model.py: two orbitals = two patch-averaged
# Fermi-surface points, a static Hubbard U that is local on the two *rotated* cluster
# sites, and a retarded full S.S interaction (longitudinal and transverse) that is also
# written in the site basis.
#
#   mpirun -n <N> python cthyb_dca_spin_spin.py --J_inter 0.5 --n_cycles 1000000
#
# Run check_rotation.py first -- it validates the rotation, the expansion and the sign
# conventions symbolically in about a second, with no core hours.
#
# What is being tested that nothing else in benchmark/dynamic_int tests:
#   * The dynamical interaction is off-diagonal in the solver's working basis. Every
#     previous dynamical-interaction benchmark here couples densities or spin-flips that
#     are diagonal in the basis the solver works in; here the vertices are the rotated
#     bilinears c^dag_{K sigma} c_{K' sigma'} with K != K', so the stochastic double
#     expansion (moves/insert_dyn.cpp, remove_dyn.cpp, swap_dyn.cpp) is doing the general
#     4-index work it was written for.
#   * The working-basis densities n_{K sigma} do NOT commute with h_loc (check_rotation.py
#     confirms this), so the plain per-vertex Lang-Firsov test rejects everything. Only
#     the part of the density-density coupling lying in span{N_up, N_down} can be
#     absorbed analytically, via the projector split in split_density_couplings. With
#     --lang_firsov False that path is switched off entirely and every vertex is sampled,
#     which is the independent route to the same answer.
#
# Observables:
#   G_tau                 the patch-resolved Green function.
#   O_tau                 <S_tot^z(tau) S_tot^z(0)>, the cluster's uniform spin
#                         correlator. S_tot^z is the only spin operator admissible here:
#                         measure_O_tau requires [O, h_loc] = 0, which the site-basis and
#                         patch-basis spin operators both fail (the site ones because the
#                         two patch levels differ, the patch ones because U is local on
#                         the sites). check_rotation.py checks exactly this.
#   dyn_vertex_corr_tau   the coupling-derivative estimator, <op1(tau) op2(0)> for every
#                         stochastic vertex. Contracting these with the expansion
#                         coefficients rebuilds the site-resolved <S_i(tau).S_j(0)>, which
#                         O_tau cannot reach -- see plot_dca_spin_spin.py. Only the
#                         stochastic vertices are measured, so the site-resolved
#                         reconstruction needs --lang_firsov False to be complete.

import argparse
import os
import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import Fourier
from triqs.operators import c, c_dag
from h5 import HDFArchive
from triqs_cthyb import Solver
import dca_model as model_def
from dca_model import key_to_string, N_PATCH

str_to_bool = lambda x: str(x).lower() in ['true', '1', 'yes']
parser = argparse.ArgumentParser(description='CTHYB: two-patch DCA with a real-space retarded S.S interaction.')
model_def.add_model_args(parser)
parser.add_argument('--n_cycles', type=int, default=1000000)
parser.add_argument('--n_warmup_cycles', type=int, default=50000)
parser.add_argument('--length_cycle', type=int, default=100)
parser.add_argument('--dyn_n_l', type=int, default=50, help='Legendre coefficients for the dynamical interaction')
parser.add_argument('--lang_firsov', type=str_to_bool, default=True,
                    help='False forces every vertex through the stochastic expansion, including the part of the '
                         'density coupling that lies in span{N_up, N_down}. Needed for the site-resolved '
                         'correlator reconstruction, and the independent cross-check of the analytic path')
parser.add_argument('--measure_O_tau_min_ins', type=int, default=100)
parser.add_argument('--random_seed', type=int, default=None,
                    help='Base seed; rank r uses base + 928374 * r. The solver default is fixed, so a plain '
                         'rerun is bit-identical -- pass different values for independent samples')
parser.add_argument('--dry_run', type=str_to_bool, default=False,
                    help='Build the model, the hybridization and the full vertex list, print them, and stop '
                         'before any Monte Carlo. Use this to validate a parameter set for free')
parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
args = parser.parse_args()
M = model_def.Model(args)

n_iw, n_tau, n_tau_bosonic = 1025, 4096, 2001
S = Solver(beta=M.beta, gf_struct=M.gf_struct, n_iw=n_iw, n_tau=n_tau,
           n_tau_bosonic=n_tau_bosonic, delta_interface=True)
for bl, delta in M.delta_iw(n_iw):
    S.Delta_tau[bl] << Fourier(delta)

# Expand the site-basis S.S into single-bilinear monomial pairs and register each one.
# add_dyn_vertex ignores any coefficient on its operator arguments, so every numeric
# factor rides in the coupling -- dca_model.register_vertices does that by construction.
registered = M.register_vertices(S, n_tau_bosonic, basis='site')
if mpi.is_master_node():
    print(f"Registered {len(registered)} dynamical vertices from the site-basis S.S "
          f"(J_intra = {M.J_intra}, J_inter = {M.J_inter}), patch levels {np.round(M.eps_patch, 5)}")

if args.dry_run:
    if mpi.is_master_node():
        print(f"\nmu = {M.mu}, h_int = {M.h_int()}")
        print(f"h_loc0 = {M.h_loc0()}")
        print(f"\nDelta_tau[up][0,0](0) = {S.Delta_tau['up'].data[0, 0, 0].real:+.6f}, "
              f"[1,1](0) = {S.Delta_tau['up'].data[0, 1, 1].real:+.6f}  "
              f"(off-diagonal max {abs(S.Delta_tau['up'].data[:, 0, 1]).max():.2e}, must be 0)")
        print(f"\n{len(registered)} vertices, coefficients multiply Q(tau):")
        for op1, op2, coeff in registered:
            print(f"  {coeff:+.5f}   ({op1}) (tau) * ({op2}) (0)")
    raise SystemExit(0)

solve_params = dict(
    h_int=M.h_int(),
    n_cycles=args.n_cycles,
    n_warmup_cycles=args.n_warmup_cycles,
    length_cycle=args.length_cycle,
    lang_firsov=args.lang_firsov,
    dyn_n_l=args.dyn_n_l,
    measure_pert_order=True,
    measure_D0_corr=True,
    measure_O_tau=(M.Sz_total, M.Sz_total),
    measure_O_tau_min_ins=args.measure_O_tau_min_ins,
    measure_density_matrix=args.density_matrix,
    use_norm_as_weight=args.density_matrix,
    **({} if args.random_seed is None else dict(random_seed=args.random_seed + 928374 * mpi.rank)))

# mu. The instantaneous part of this retarded interaction is, exactly,
#     H_shift = sum_v K'_v(0) op1_v op2_v = (1/(2 omega_0^2)) sum_ij (-J_ij) S_i.S_j
# (K'_v(0) = coeff_v/(2 omega_0^2) in closed form; the second equality is the sum rule
# check_rotation.py asserts). That is a pure SPIN bilinear, so it is particle-hole even and
# contributes exactly NOTHING to the filling condition -- verified symbolically, zero residual.
# The patch levels are exactly antisymmetric (each patch is exactly half the zone), and the
# site-basis U is particle-hole even, so for a symmetric bath mu = U/2 is exactly half filling.
#
# Deliberately NOT derived from a probe solve. lang_firsov_U_renorm / lang_firsov_mu_renorm
# report only what the solver routed analytically, so they are route dependent: taking mu from
# them would give the lang_firsov=True and lang_firsov=False runs different Hamiltonians and
# invalidate the cross-check that is the whole point of running the pair. See dyn_static_shift.py
# for the general (e.g. Holstein) case, where the offset is nonzero and must be folded in.
mu_orbital = np.full(len(M.labels), M.mu)
if not M.half_filling_is_exact() and mpi.is_master_node():
    print(f"NOTE: --bath {M.bath} breaks particle-hole symmetry (the two patch densities of states "
          f"have different widths, variance ratio {M.params()['dos_variance_ratio']:.3f}), so "
          f"mu = {M.mu} is only approximately half filling. Read the measured filling below and "
          f"tune --mu against it; this is why vbdmft.py carries a doping -> mu table. "
          f"--bath discrete is particle-hole symmetric and needs no tuning.")
h_loc0 = sum((M.eps_patch[K] - mu_orbital[M.labels.index((s, K))]) * c_dag(s, K) * c(s, K)
             for s in model_def.SPIN_NAMES for K in range(N_PATCH))

S.solve(**solve_params, h_loc0=h_loc0)

if mpi.is_master_node():
    # Densities, so the filling is visible without any post-processing
    fillings = {bl: [-g.data[-1, i, i].real for i in range(N_PATCH)] for bl, g in S.G_tau}
    total_filling = sum(sum(v) for v in fillings.values())
    print(f"average sign {S.average_sign}, average order {S.average_order}")
    print(f"patch fillings {dict((k, np.round(v, 5).tolist()) for k, v in fillings.items())}, "
          f"total <N> = {total_filling:.5f} (2.0 is half filling)")

    # Label every measured vertex correlator by its monomial pair, so the analysis script
    # can contract them into site-resolved <S_i(tau).S_j(0)> without relying on ordering.
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
                            f"cthyb_dca_{M.tag()}_lf-{args.lang_firsov}_nc-{args.n_cycles}{seed_tag}.h5")
    with HDFArchive(filename, 'w') as A:
        A['G_tau'] = S.G_tau
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
        A['mu_orbital'] = mu_orbital
        A['lang_firsov'] = args.lang_firsov
        A['n_cycles'] = args.n_cycles
        A['random_seed'] = -1 if args.random_seed is None else args.random_seed
    print(f"Saved {filename}")
