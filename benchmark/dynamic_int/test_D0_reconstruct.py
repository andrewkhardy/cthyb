import numpy as np
import scipy.special as special

def build_M_matrix(N, beta):
    M = np.zeros((N, N))
    beta_sq = beta**2
    if N > 0:
        M[0, 0] = -beta_sq / 12.0
    if N > 1:
        M[1, 1] = -beta_sq / 60.0
    for q in range(N):
        if q >= 2:
            M[q, q] = -beta_sq / (2.0 * (2*q - 1) * (2*q + 3))
        if q + 2 < N:
            M[q + 2, q] = beta_sq / (4.0 * (2*q + 1) * (2*q + 3))
        if q - 2 >= 0:
            M[q - 2, q] = beta_sq / (4.0 * (2*q - 1) * (2*q + 1))
    return M

beta = 10.0
N_leg = 30

# Let us consider a single segment from tau_start to tau_end
# n(tau) is 1 inside [tau_start, tau_end], 0 outside.
# Let tau_start = 2.0, tau_end = 6.0.
# So the operators are:
# op1: c^\dagger at 2.0 (dagger=True, S = +1)
# op2: c at 6.0 (dagger=False, S = -1)
ops = [(2.0, True), (6.0, False)]

# The exact density n(tau) for this sample is:
def n_exact(tau):
    return 1.0 if (2.0 <= tau <= 6.0) else 0.0

# The exact density-density correlator for this single sample is:
# Q(tau) = \frac{1}{\beta} \int_0^\beta n(tau_1 + tau) n(tau_1) dtau_1
# Since it is periodic, let us compute it on a grid.
n_tau = 1000
tau_grid = np.linspace(0, beta, n_tau)
Q_exact = np.zeros(n_tau)
dtau_int = beta / n_tau
for i, tau in enumerate(tau_grid):
    val = 0.0
    for j in range(n_tau):
        tau1 = j * dtau_int
        tau2 = (tau1 + tau) % beta
        val += n_exact(tau1) * n_exact(tau2) * dtau_int
    Q_exact[i] = val / beta

# Let us project Q_exact to Legendre coefficients q_n_exact
q_n_exact = np.zeros(N_leg)
for n in range(N_leg):
    x = 2.0 * tau_grid / beta - 1.0
    P_n = special.eval_legendre(n, x)
    # q_n = (2n+1)/beta * \int_0^\beta Q(tau) P_n(x) dtau
    q_n_exact[n] = (2.0 * n + 1.0) / beta * np.sum(Q_exact * P_n) * (beta / (n_tau - 1))

# Now let us compute alpha_n from the operators!
# alpha_n = \sum_{i,j} S_i S_j P_n(x_ij)
# Where S_i = +1 for dagger=True, -1 for dagger=False.
# Let us include both self-pairs and cross-pairs.
alpha_n = np.zeros(N_leg)
# Self pairs:
# i=0: S_0 = 1, x_00 = -1 (dt=0 mapped to -1)
# i=1: S_1 = -1, S_1^2 = 1, x_11 = -1
for n in range(N_leg):
    alpha_n[n] += special.eval_legendre(n, -1.0) * 1.0 # op0 self
    alpha_n[n] += special.eval_legendre(n, -1.0) * 1.0 # op1 self

# Cross pairs:
# (0,1): S_0 = 1, S_1 = -1, dt = 2.0 - 6.0 = -4.0 -> wrapped to 6.0
# (1,0): S_1 = -1, S_0 = 1, dt = 6.0 - 2.0 = 4.0 -> wrapped to 4.0
dt_01 = 2.0 - 6.0
if dt_01 < 0: dt_01 += beta
x_01 = 2.0 * dt_01 / beta - 1.0

dt_10 = 6.0 - 2.0
if dt_10 < 0: dt_10 += beta
x_10 = 2.0 * dt_10 / beta - 1.0

for n in range(N_leg):
    alpha_n[n] += (1.0 * -1.0) * special.eval_legendre(n, x_01)
    alpha_n[n] += (-1.0 * 1.0) * special.eval_legendre(n, x_10)

# Now let us reconstruct using M.
M = build_M_matrix(N_leg, beta)

# Test 1: Code's formula (no minus sign, M transpose)
# q_n = sum_p M(p, n) * alpha_p * (2n+1) / beta^2
q_n_test1 = np.zeros(N_leg)
for n in range(N_leg):
    s = 0.0
    for p in range(N_leg):
        s += M[p, n] * alpha_n[p]
    q_n_test1[n] = s * (2.0 * n + 1.0) / (beta * beta)

# Test 2: Formula with minus sign:
# q_n = - sum_p M(p, n) * alpha_p * (2n+1) / beta^2
q_n_test2 = -q_n_test1

# Test 3: Using M matrix directly (no transpose, i.e., M @ alpha_n):
q_n_test3 = np.zeros(N_leg)
for n in range(N_leg):
    s = 0.0
    for p in range(N_leg):
        s += M[n, p] * alpha_n[p]
    q_n_test3[n] = -s * (2.0 * n + 1.0) / (beta * beta)

# Test 4: Pure M @ alpha
q_n_test4 = M @ alpha_n

# Test 5: True alpha
alpha_true = (2.0 * np.arange(N_leg) + 1.0) / beta * alpha_n
q_n_test5 = M @ alpha_true

# Test 6: Code's formula but divided by beta instead of beta^2
q_n_test6 = np.zeros(N_leg)
for n in range(N_leg):
    s = 0.0
    for p in range(N_leg):
        s += M[p, n] * alpha_n[p]
    q_n_test6[n] = s * (2.0 * n + 1.0) / beta

print("Exact q_0:", q_n_exact[0])
print("Test 1 q_0:", q_n_test1[0])
print("Test 2 q_0:", q_n_test2[0])
print("Test 3 q_0:", q_n_test3[0])
print("Test 4 q_0:", q_n_test4[0])
print("Test 5 q_0:", q_n_test5[0])
print("Test 6 q_0:", q_n_test6[0])

print("\nExact q_1:", q_n_exact[1])
print("Test 1 q_1:", q_n_test1[1])
print("Test 2 q_1:", q_n_test2[1])
print("Test 3 q_1:", q_n_test3[1])
print("Test 4 q_1:", q_n_test4[1])
print("Test 5 q_1:", q_n_test5[1])
print("Test 6 q_1:", q_n_test6[1])

# Let's check the error for all N_leg
err1 = np.max(np.abs(q_n_exact - q_n_test1))
err2 = np.max(np.abs(q_n_exact - q_n_test2))
err3 = np.max(np.abs(q_n_exact - q_n_test3))
err4 = np.max(np.abs(q_n_exact - q_n_test4))
err5 = np.max(np.abs(q_n_exact - q_n_test5))
err6 = np.max(np.abs(q_n_exact - q_n_test6))
print("\nMax errors:")
print("Test 1 (code):", err1)
print("Test 2 (-code):", err2)
print("Test 3 (M @ alpha):", err3)
print("Test 4 (Pure M @ alpha):", err4)
print("Test 5 (M @ alpha_true):", err5)
print("Test 6 (beta normalization):", err6)
