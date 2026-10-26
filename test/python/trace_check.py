import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import *
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure
from triqs.operators.util.observables import S_op
from triqs_cthyb import *

# trace_check on a sign-free model (move_global's), with every operator flipped by the global move: the solve must
# run through the checks after every accept and reject, print its report, and keep Tr rho = 1

beta = 60.0
n_orb = 2
mu = 1.0
U = 2.0
J = 0.2
h = 0.1
epsilon = 2.3
V = 2.0 * np.eye(n_orb) + 0.2 * (np.ones(n_orb) - np.eye(n_orb))

spin_names = ('up', 'dn')
gf_struct = set_operator_structure(spin_names, n_orb, True)

S = Solver(beta=beta, gf_struct=gf_struct, n_iw=1025, n_tau=2500, n_l=50)

H = h_int_kanamori(spin_names, n_orb,
                   np.array([[0, U - 3 * J], [U - 3 * J, 0]]),
                   np.array([[U, U - 2 * J], [U - 2 * J, U]]),
                   J, off_diag=True)
H += h * S_op('z', spin_names, n_orb, True)

delta_w = GfImFreq(beta=beta, target_shape=[n_orb, n_orb])
delta_w << inverse(iOmega_n - epsilon) + inverse(iOmega_n + epsilon)
delta_w.from_L_G_R(V, delta_w, V)
S.G0_iw << inverse(iOmega_n + mu - delta_w)

p = {}
p["random_seed"] = 123 * mpi.rank + 567
p["length_cycle"] = 50
p["n_warmup_cycles"] = 500
p["n_cycles"] = 2000
p["move_double"] = False
p["move_global"] = {'flip_spins': {("up", 0): ("dn", 0), ("dn", 0): ("up", 0), ("up", 1): ("dn", 1), ("dn", 1): ("up", 1)}}
p["move_global_prob"] = 0.06
p["move_global_full"] = True
p["measure_density_matrix"] = True
p["use_norm_as_weight"] = True
p["trace_check"] = True

S.solve(h_int=H, **p)

tr = sum(np.trace(np.asarray(b)) for b in S.density_matrix)
assert abs(tr - 1) < 1e-8, f"Tr rho = {tr}"
