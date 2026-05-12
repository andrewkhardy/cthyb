import numpy as np
import scipy.integrate as integrate
from scipy.special import eval_legendre

beta = 10.0
w_c = 1.0
x = 0.5

def D0(tau):
    return -x**2 * np.cosh(w_c * (tau - beta / 2.0)) / np.sinh(beta * w_c / 2.0)

coeffs = []
for n in range(50):
    val, _ = integrate.quad(lambda tau: D0(tau) * eval_legendre(n, 2.0*tau/beta - 1.0), 0, beta)
    val *= (2.0 * n + 1.0) / beta
    coeffs.append(val)

for i in range(16):
    print(f"n={i}: {coeffs[i]:.3e}")
