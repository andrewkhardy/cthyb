import numpy as np
from triqs.gf import *
from triqs.gf.descriptors import Function
from triqs.gf.tools import *
from triqs.gf.block_gf import *

J = -0.1
w0 = 0.1
beta = 10.0
hopping = 1.0
U = 4.0
n_tau_bosonic = 2001
n_iw = 1025

try:
    q_tau = GfImTime(indices=[0],  statistic='Boson', beta=beta, n_points=n_tau_bosonic)
    q_iw = make_gf_from_fourier(q_tau)  
    q_iw << Function(lambda w: 2 * J/w0 * w0**2 / (w**2 - w0**2))
    q_tau = make_gf_from_fourier(q_iw)
    Q_tau = Block2Gf(['up', 'down'], ['up', 'down'], [[q_tau, q_tau], [q_tau, q_tau]])
    Q_iw = make_gf_from_fourier(Q_tau)
    
    g0 = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
    g0 << SemiCircular(2*hopping)
    
    ivn = np.array([x.imag for x in Q_iw["up", "up"].mesh.values()])
    # In python 3 / TRIQS, Q_iw.mesh.values() might not be directly iterable or imaginary part is extracted differently.
    # We will test this.
    zero_freq = np.where(np.abs(ivn) < 1e-10)
    print(ivn[zero_freq])
    mu = U/2 - np.real((Q_iw["up", "up"].data[zero_freq][0,0,0]+Q_iw["up", "down"].data[zero_freq][0,0,0])/2.0)
    
    print("Successfully calculated mu:", mu)
    print("Zero freq index:", zero_freq)
except Exception as e:
    import traceback
    traceback.print_exc()
