import numpy as np
import scipy.integrate as integrate
from scipy.special import eval_legendre

beta = 10.0
g02 = -0.35
wp = 0.35

def D0(tau):
    # D(iw) = 2 g02 wp / ( (iw)^2 - wp^2 )
    # D(tau) = - g02 * cosh(wp*(tau - beta/2)) / sinh(wp*beta/2)
    return -g02 * np.cosh(wp * (tau - beta / 2.0)) / np.sinh(beta * wp / 2.0)

coeffs = []
for n in range(50):
    val, _ = integrate.quad(lambda tau: D0(tau) * eval_legendre(n, 2.0*tau/beta - 1.0), 0, beta)
    val *= (2.0 * n + 1.0) / beta
    coeffs.append(val)

for i in range(25):
    print(f"n={i}: {coeffs[i]:.3e}")
