from triqs.gfs import *
import argparse
import triqs.utility.mpi as mpi
from triqs.gf.descriptors import Function
from triqs.gf.tools import *
from triqs.gf.block_gf import *
from triqs.operators import n
import h5
from triqs.utility.h5diff import h5diff
from triqs_ctseg import Solver
# Parse command line arguments
parser = argparse.ArgumentParser(description='Run spin-spin benchmarking.')
parser.add_argument('--U', type=float, default=4.0, help='U parameter')
parser.add_argument('--L', type=float, default=1.0, help='L parameter')
parser.add_argument('--beta', type=float, default=10.0, help='Inverse temperature') 
parser.add_argument('--n_cycles', type=int, default=1000000, help='Number of MC cycles')
args, unknown = parser.parse_known_args()

# Numerical values
hopping = 1.0
w0 = 0.1
beta = args.beta
U = args.U

L = args.L
n_tau = 4096+3
n_tau_bosonic = 3999
n_iw = 2048
h_int = U*n("up", 0)*n("down", 0)
# Solver construction parameters
gf_struct = [('down', 1), ('up', 1)]
constr_params = {
    "gf_struct": gf_struct,
    "beta": beta,
    "n_tau": n_tau,
    "n_tau_bosonic": n_tau_bosonic
}

# Construct solver
S = Solver(**constr_params)



# Hybridization Delta(tau)
Delta = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
g0 = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
q_tau = GfImTime(indices=[0],  statistic='Boson', beta=beta, n_points=n_tau_bosonic)
q_iw = make_gf_from_fourier(q_tau)  
q_iw << Function(lambda w: 2 * L/w0 * w0**2 / (w**2 - w0**2))
q_tau << Fourier(q_iw)
Q_tau = Block2Gf(['up', 'down'], ['up', 'down'], [[q_tau, q_tau], [q_tau, q_tau]])
Q_iw = make_gf_from_fourier(Q_tau)
g0 << SemiCircular(2*hopping)
ivn = np.array([x.imag for x in Q_iw["up", "up"].mesh.values()])
zero_freq = np.where(np.abs(ivn) < 1e-10)
mu = U/2 + np.real((Q_iw["up", "up"].data[zero_freq][0,0,0]+Q_iw["up", "down"].data[zero_freq][0,0,0]))/2.0
#Delta <<  inverse(g0) - iOmega_n - mu
print(f"Zero frequency point: {Q_iw['up', 'up'].data[zero_freq][0,0,0]}")

Delta << hopping**2*SemiCircular(2*hopping)
S.Delta_tau << Fourier(Delta)
S.D0_tau << Q_tau


# Solve parameters
solve_params = {
    "h_int": h_int,
    "h_loc0": -mu * (n("up", 0) + n("down", 0)),
    "length_cycle": 100,
    "n_warmup_cycles": 100000,
    "n_cycles": args.n_cycles,
    "measure_F_tau": True,
    "measure_nn_tau": True,
    #"measure_nn_nu": True,
    "measure_nn_static": True,
    "measure_pert_order": True,
    "lang_firsov": True
    }

# Solve
S.solve(**solve_params)
print(S.results.F_tau)
print(S.results.nn_tau)
print(S.results.nn_nu)
print(S.results.nn_static)
print(S.results.densities)
print(S.results.average_sign)
print(S.results.pert_order_Delta)
# Save data
if mpi.is_master_node():
    filename = f"/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin_ctseg_lambda-{L}-U-{U}_b-{beta}.h5"
    with h5.HDFArchive(filename, "w") as A:
        A['G_tau'] = S.results.G_tau
        A['F_tau'] = S.results.F_tau
        A['nn_tau'] = S.results.nn_tau
        A['nn_nu'] = S.results.nn_nu
        A['nn'] = S.results.nn_static
        A['densities'] = S.results.densities
        A["average_sign"] = S.results.average_sign
        A["perturbation_order_D"] = S.results.pert_order_Delta

