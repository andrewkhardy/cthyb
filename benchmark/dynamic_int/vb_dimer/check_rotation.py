# Internal consistency checks for the two-patch DCA / real-space S.S setup in model.py.
# Everything here is symbolic or linear algebra -- no Monte Carlo, no solver, runs in a
# second -- so it can be run before committing any core hours.
#
#   python check_rotation.py                      (defaults: J_intra 0, J_inter 0.5)
#   python check_rotation.py --J_intra 0.5 --J_inter 0.5    (the uniform case)
#
# The checks, in order:
#
# 1. CONVENTION. With --rotation none the site basis *is* the working basis, so the
#    expanded vertex list must reproduce the coefficients of spin_spin.py -- the
#    validated single-orbital full-S.S benchmark -- term for term:
#       (n_up, n_up) = -J/4,  (n_up, n_down) = +J/4,  (S^+, S^-) = -J/2   (times Q(tau))
#    This pins the sign and factor conventions to a script that already agrees with
#    CT-INT, instead of re-deriving them.
#
# 2. BASIS INDEPENDENCE. sum_i c^dag_{i s} c_{i s'} is the trace of c^dag c over the site
#    index and the rotation R is orthogonal, so S_tot = sum_i S_i = sum_K S_K exactly.
#    Therefore in the *uniform* case (J_intra == J_inter) the retarded interaction is
#    lambda(tau) S_tot(tau).S_tot(0), which is the same object whether it is built by
#    summing over site pairs or over patch pairs. The two expansions must produce
#    *identical* merged vertex lists. This is an exact, zero-cost unit test of the
#    rotation and of expand_retarded_product: any error in R, in the monomial bookkeeping
#    or in the merging shows up as a mismatch.
#
# 3. COMMUTATORS -- who is conserved. Confirms the premise of the whole exercise:
#    the site-basis densities commute with the Hubbard U (it is local in that basis) but
#    NOT with the patch levels, the working-basis densities n_{K sigma} do not commute
#    with U, and N_up / N_down / N_total and S_tot^z commute with everything. The second
#    of these is what forces the dynamical interaction off the plain Lang-Firsov path;
#    the last is what makes S_tot^z admissible for measure_O_tau (which requires
#    [O, h_loc] = 0).
#
# 4. VERTEX CENSUS. How many monomial-pair vertices the run will actually carry, and
#    whether the list is closed under swapping op1 and op2 (the moves assume it is).
#
# 5. HERMITICITY / SUM RULE. The expanded interaction, re-assembled as an operator at
#    equal times, must equal sum_{ij} lambda_ij S_i.S_j evaluated as a plain operator
#    product -- a check that no monomial was lost or double counted.

import argparse
import numpy as np
from triqs.operators import c, c_dag, Operator, dagger
import model as model_def
from model import SPIN_NAMES, N_PATCH, expand_retarded_product, key_to_operator, prune

parser = argparse.ArgumentParser(description='Symbolic checks for the DCA real-space S.S setup.')
model_def.add_model_args(parser)
args = parser.parse_args()

failures = []


def report(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ''))
    if not ok:
        failures.append(name)


def is_zero(op, tol=1e-12):
    return all(abs(coeff) < tol for _, coeff in op)


def max_coeff(op):
    return max((abs(coeff) for _, coeff in op), default=0.0)


# =======================================================================================
print("\n1. Convention anchor against spin_spin.py (rotation = identity)")
# =======================================================================================
J = 1.0
unrotated = argparse.Namespace(**vars(args))
unrotated.rotation = 'none'
unrotated.J_intra, unrotated.J_inter = J, 0.0
M0 = model_def.Model(unrotated)
v0 = M0.spin_spin_vertices('site')


def coeff_of(vertices, op1, op2):
    """Look up the merged coefficient of the (op1, op2) monomial pair."""
    def key(op):
        (mono, _), = list(op)
        return tuple((bool(d), tuple(i)) for d, i in mono)
    return vertices.get((key(op1), key(op2)), 0.0)


# Site 0 with the identity rotation is patch 0; spin_spin.py's single orbital.
n_up, n_dn = c_dag('up', 0) * c('up', 0), c_dag('down', 0) * c('down', 0)
Sp, Sm = c_dag('up', 0) * c('down', 0), c_dag('down', 0) * c('up', 0)
expected = {
    'D0(up, up)     vs spin_spin.py -0.25*J': (coeff_of(v0, n_up, n_up), -0.25 * J),
    'D0(down, down) vs spin_spin.py -0.25*J': (coeff_of(v0, n_dn, n_dn), -0.25 * J),
    'D0(up, down)   vs spin_spin.py +0.25*J': (coeff_of(v0, n_up, n_dn), +0.25 * J),
    'D0(down, up)   vs spin_spin.py +0.25*J': (coeff_of(v0, n_dn, n_up), +0.25 * J),
    'Jperp(S+, S-)  vs spin_spin.py -0.50*J': (coeff_of(v0, Sp, Sm), -0.50 * J),
    'Jperp(S-, S+)  vs spin_spin.py -0.50*J': (coeff_of(v0, Sm, Sp), -0.50 * J),
}
for name, (got, want) in expected.items():
    report(name, abs(got - want) < 1e-14, f"got {got:+.6f}, expected {want:+.6f}")

# The relation that makes longitudinal and transverse parts describe the *same* coupling.
# lambda S^z S^z puts lambda/4 on (n_s, n_s), while expand_Jperp_into_vertices consumes
# Jperp_tau/2 per ordering of lambda/2 (S^+S^- + S^-S^+). So an SU(2)-symmetric S.S needs
#     Jperp_tau == 4 * D0_same-spin,  with the same sign.
# spin_spin.py satisfies it (D0[up,up] = -0.25 J Q, Jperp = -J Q). The deleted
# multiorb_spin_spin.py had D0[up,up] = +0.25 J Q against Jperp = -J Q, i.e. its two parts
# described opposite couplings -- which is why this is checked rather than commented.
jperp_tau = 2.0 * coeff_of(v0, Sp, Sm)
report("SU(2) consistency: Jperp_tau == 4 * D0_same-spin",
       abs(jperp_tau - 4.0 * coeff_of(v0, n_up, n_up)) < 1e-14,
       f"Jperp_tau = {jperp_tau:+.6f}, 4 * D0_same-spin = {4.0 * coeff_of(v0, n_up, n_up):+.6f}")

# =======================================================================================
print("\n2. Basis independence: uniform J gives S_tot.S_tot in either basis")
# =======================================================================================
uniform = argparse.Namespace(**vars(args))
uniform.rotation = 'site'
uniform.J_intra = uniform.J_inter = 0.7
Mu = model_def.Model(uniform)
v_site = Mu.spin_spin_vertices('site')
v_patch = Mu.spin_spin_vertices('patch')

keys = set(v_site) | set(v_patch)
worst = max((abs(v_site.get(k, 0.0) - v_patch.get(k, 0.0)) for k in keys), default=0.0)
report("site-pair sum == patch-pair sum (uniform J)", worst < 1e-12,
       f"{len(v_site)} vs {len(v_patch)} vertices, max |difference| = {worst:.2e}")

# S_tot really is basis independent, as an operator
report("sum_i S_i^z == sum_K S_K^z", is_zero(sum(Mu.Sz_site) - sum(Mu.Sz_patch)),
       f"max |residual coeff| = {max_coeff(sum(Mu.Sz_site) - sum(Mu.Sz_patch)):.2e}")
report("sum_i S_i^+ == sum_K S_K^+", is_zero(sum(Mu.Sp_site) - sum(Mu.Sp_patch)),
       f"max |residual coeff| = {max_coeff(sum(Mu.Sp_site) - sum(Mu.Sp_patch)):.2e}")

# ... and the non-uniform case must NOT be basis independent, or the test above is vacuous
nonuniform = argparse.Namespace(**vars(args))
nonuniform.rotation = 'site'
nonuniform.J_intra, nonuniform.J_inter = 0.0, 0.7
Mn = model_def.Model(nonuniform)
d_site, d_patch = Mn.spin_spin_vertices('site'), Mn.spin_spin_vertices('patch')
spread = max((abs(d_site.get(k, 0.0) - d_patch.get(k, 0.0)) for k in set(d_site) | set(d_patch)), default=0.0)
report("non-uniform J is basis DEPENDENT (control)", spread > 1e-6,
       f"max |difference| = {spread:.3f} -- the rotation is doing real work")

# =======================================================================================
print("\n3. Commutators with h_loc: which densities are conserved")
# =======================================================================================
M = model_def.Model(args)
h_loc, h_int = M.h_loc(), M.h_int()


def commutes(op, with_op):
    return is_zero(op * with_op - with_op * op)


for i in range(N_PATCH):
    report(f"[n_site{i + 1},up , h_int] == 0 (U is local on the sites)",
           commutes(M.n_site[(i, 'up')], h_int))
for K in range(N_PATCH):
    n_K = c_dag('up', K) * c('up', K)
    report(f"[n_patch{K} ,up , h_int] != 0 (working basis is NOT conserved)",
           not commutes(n_K, h_int))
report("[N_up, h_loc] == 0", commutes(sum(c_dag('up', K) * c('up', K) for K in range(N_PATCH)), h_loc))
report("[N_total, h_loc] == 0", commutes(M.N_total, h_loc))
report("[S_tot^z, h_loc] == 0  (so measure_O_tau accepts it)", commutes(M.Sz_total, h_loc))
if abs(M.eps_patch[0] - M.eps_patch[1]) > 1e-10:
    report("[n_site1,up , h_loc] != 0 (patch levels differ, so site densities are not conserved either)",
           not commutes(M.n_site[(0, 'up')], h_loc),
           f"eps_patch = {np.round(M.eps_patch, 5)}")
    report("[S_patch^z, h_loc] != 0 (so it is NOT admissible for measure_O_tau)",
           not commutes(M.Sz_patch[0], h_loc))

# =======================================================================================
print("\n4. Vertex census for the requested parameters")
# =======================================================================================
v = M.spin_spin_vertices('site')
print(f"   J_intra = {M.J_intra}, J_inter = {M.J_inter}, rotation = {M.rotation_name}")
n_raw = sum(len(list(a)) * len(list(b))
            for i in range(N_PATCH) for j in range(N_PATCH)
            if (M.J_intra if i == j else M.J_inter) != 0.0
            for a, b in ((M.Sz_site[i], M.Sz_site[j]), (M.Sp_site[i], M.Sm_site[j]), (M.Sm_site[i], M.Sp_site[j])))
print(f"   {n_raw} raw monomial pairs -> {len(v)} vertices after merging duplicates and "
      f"pruning cancellations")
swapped_missing = [k for k in v if (k[1], k[0]) not in v]
report("vertex list closed under swapping op1 <-> op2", not swapped_missing,
       f"{len(swapped_missing)} unmatched" if swapped_missing else "the moves rely on this")
asym = max((abs(v[k] - v[(k[1], k[0])]) for k in v if (k[1], k[0]) in v), default=0.0)
report("coefficients symmetric under the swap", asym < 1e-12, f"max asymmetry = {asym:.2e}")

density_like = sum(1 for (k1, k2) in v if k1[0][1] == k1[1][1] and k2[0][1] == k2[1][1])
print(f"   of these, {density_like} are density-density (n_a, n_b) pairs -- the only shape the "
      f"Lang-Firsov\n   projector split can absorb; the other {len(v) - density_like} are "
      f"genuinely off-diagonal and\n   must go through insert_dyn/remove_dyn.")

# =======================================================================================
print("\n5. Sum rule: the expansion reassembles the original operator")
# =======================================================================================
# At equal times the retarded interaction collapses to an ordinary operator product, so
# summing coeff * op1 * op2 over the expanded vertices must reproduce
# sum_{ij} -J_ij S_i.S_j computed directly. This catches a lost or duplicated monomial.
direct = Operator()
for i in range(N_PATCH):
    for j in range(N_PATCH):
        J_ij = M.J_intra if i == j else M.J_inter
        if J_ij == 0.0:
            continue
        direct += -J_ij * (M.Sz_site[i] * M.Sz_site[j]
                           + 0.5 * (M.Sp_site[i] * M.Sm_site[j] + M.Sm_site[i] * M.Sp_site[j]))
reassembled = Operator()
for (k1, k2), coeff in v.items():
    reassembled += coeff * key_to_operator(k1) * key_to_operator(k2)
residual = direct - reassembled
report("sum_v coeff * op1 * op2 == sum_ij -J_ij S_i.S_j", is_zero(residual, 1e-10),
       f"max |residual coeff| = {max_coeff(residual):.2e}")

# =======================================================================================
print(f"\n{'All checks passed.' if not failures else f'{len(failures)} CHECK(S) FAILED: ' + ', '.join(failures)}\n")
raise SystemExit(1 if failures else 0)
