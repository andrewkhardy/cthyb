# Symbolic checks of the vb_dimer model and of how run_cthyb.py's add_dyn_int calls expand.
# No Monte Carlo; runs in about a second.
#
#   python check_rotation.py
#   python check_rotation.py --J_intra 0.5 --J_inter 0.5
#
# 1. Convention: with --rotation none, the expansion reproduces spin_spin.py's validated
#    D0_tau / Jperp_tau coefficients.
# 2. Basis independence: for uniform J the interaction is S_tot.S_tot, so summing over site
#    pairs or patch pairs gives the same vertices.
# 3. Commutators: which densities commute with h_loc.
# 4. Vertex census, and symmetry of the vertex list under op1 <-> op2.
# 5. Sum rule: the vertices reassemble sum_ij -J_ij S_i.S_j at equal times.

import argparse
from itertools import product

from triqs.operators import Operator, c, c_dag
from triqs_cthyb.dynamical_interactions import bilinear, expand_dyn_int

import model as model_def
from model import N_PATCH

parser = argparse.ArgumentParser(description='Symbolic checks for the vb_dimer model.')
model_def.add_model_args(parser)
args = parser.parse_args()

failures = []


def report(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ''))
    if not ok:
        failures.append(name)


def model(**overrides):
    a = argparse.Namespace(**vars(args))
    for k, v in overrides.items():
        setattr(a, k, v)
    return model_def.Model(a)


def max_coeff(op):
    return max((abs(coeff) for _, coeff in op), default=0.0)


def spin_spin_vertices(J, Sz, Sp, Sm):
    """The merged vertices of run_cthyb.py's add_dyn_int calls, as coefficients of Q(tau)."""
    vertices = {}
    for i, j in product(range(N_PATCH), repeat=2):
        for factor, op1, op2 in ((1.0, Sz[i], Sz[j]), (0.5, Sp[i], Sm[j]), (0.5, Sm[i], Sp[j])):
            for key, coeff in expand_dyn_int(op1, op2).items():
                vertices[key] = vertices.get(key, 0.0) - J[i, j] * factor * coeff
    return {k: v for k, v in vertices.items() if abs(v) > 1e-12}


def key(op):
    (monomial, _), = list(op)
    return tuple((bool(dagger), tuple(indices)) for dagger, indices in monomial)


# ---------------------------------------------------------------------------------------
print("\n1. Convention, against spin_spin.py (rotation = identity, J_intra = 1)")
M0 = model(rotation='none', J_intra=1.0, J_inter=0.0)
v0 = spin_spin_vertices(M0.J, M0.Sz, M0.Sp, M0.Sm)
n_up, n_dn = c_dag('up', 0) * c('up', 0), c_dag('down', 0) * c('down', 0)
Sp, Sm = c_dag('up', 0) * c('down', 0), c_dag('down', 0) * c('up', 0)
for name, op1, op2, want in [("D0[up, up]     = -J/4", n_up, n_up, -0.25), ("D0[down, down] = -J/4", n_dn, n_dn, -0.25),
                             ("D0[up, down]   = +J/4", n_up, n_dn, +0.25), ("D0[down, up]   = +J/4", n_dn, n_up, +0.25),
                             ("Jperp/2 (S+, S-) = -J/2", Sp, Sm, -0.5), ("Jperp/2 (S-, S+) = -J/2", Sm, Sp, -0.5)]:
    got = v0.get((key(op1), key(op2)), 0.0)
    report(name, abs(got - want) < 1e-14, f"got {got:+.6f}")

# ---------------------------------------------------------------------------------------
print("\n2. Basis independence for uniform J")
Ms, Mp = model(J_intra=0.7, J_inter=0.7), model(J_intra=0.7, J_inter=0.7, rotation='none')
v_site = spin_spin_vertices(Ms.J, Ms.Sz, Ms.Sp, Ms.Sm)
v_patch = spin_spin_vertices(Mp.J, Mp.Sz, Mp.Sp, Mp.Sm)
worst = max(abs(v_site.get(k, 0.0) - v_patch.get(k, 0.0)) for k in set(v_site) | set(v_patch))
report("site-pair sum == patch-pair sum", worst < 1e-12, f"{len(v_site)} vertices, max |diff| = {worst:.2e}")
report("sum_i S_i^z == sum_K S_K^z", max_coeff(Ms.Sz_total - Mp.Sz_total) < 1e-12)

Mn = model(J_intra=0.0, J_inter=0.7)
v_nonuniform = spin_spin_vertices(Mn.J, Mn.Sz, Mn.Sp, Mn.Sm)
v_nonuniform_patch = spin_spin_vertices(Mn.J, Mp.Sz, Mp.Sp, Mp.Sm)
spread = max(abs(v_nonuniform.get(k, 0.0) - v_nonuniform_patch.get(k, 0.0))
             for k in set(v_nonuniform) | set(v_nonuniform_patch))
report("non-uniform J is basis dependent (control)", spread > 1e-6, f"max |diff| = {spread:.3f}")

# ---------------------------------------------------------------------------------------
print("\n3. Commutators with h_loc")
M = model_def.Model(args)
h_loc, h_int = M.h_loc(), M.h_int()


def commutes(op, other):
    return max_coeff(op * other - other * op) < 1e-12


for i in range(N_PATCH):
    report(f"[n_site{i}_up, h_int] == 0", commutes(M.n_site(i, 'up'), h_int))
for K in range(N_PATCH):
    report(f"[n_patch{K}_up, h_int] != 0", not commutes(c_dag('up', K) * c('up', K), h_int))
report("[N_up, h_loc] == 0", commutes(sum(c_dag('up', K) * c('up', K) for K in range(N_PATCH)), h_loc))
report("[S_tot^z, h_loc] == 0 (measure_O_tau needs this)", commutes(M.Sz_total, h_loc))

# ---------------------------------------------------------------------------------------
print("\n4. Vertex census")
v = spin_spin_vertices(M.J, M.Sz, M.Sp, M.Sm)
n_density = sum(1 for k1, k2 in v if k1[0][1] == k1[1][1] and k2[0][1] == k2[1][1])
print(f"   J_intra = {M.J_intra}, J_inter = {M.J_inter}: {len(v)} vertices, {n_density} density-density")
unpaired = [k for k in v if (k[1], k[0]) not in v]
report("closed under op1 <-> op2", not unpaired, f"{len(unpaired)} unpaired" if unpaired else '')
asym = max((abs(v[k] - v[(k[1], k[0])]) for k in v if (k[1], k[0]) in v), default=0.0)
report("symmetric under op1 <-> op2", asym < 1e-12, f"max asymmetry = {asym:.2e}")

# ---------------------------------------------------------------------------------------
print("\n5. Sum rule")
direct = Operator()
for i, j in product(range(N_PATCH), repeat=2):
    direct += -M.J[i, j] * (M.Sz[i] * M.Sz[j] + 0.5 * (M.Sp[i] * M.Sm[j] + M.Sm[i] * M.Sp[j]))
reassembled = sum((coeff * bilinear(k1) * bilinear(k2) for (k1, k2), coeff in v.items()), Operator())
report("sum_v coeff op1 op2 == sum_ij -J_ij S_i.S_j", max_coeff(direct - reassembled) < 1e-10,
       f"max |residual| = {max_coeff(direct - reassembled):.2e}")

print(f"\n{'All checks passed.' if not failures else f'{len(failures)} FAILED: ' + ', '.join(failures)}\n")
raise SystemExit(1 if failures else 0)
