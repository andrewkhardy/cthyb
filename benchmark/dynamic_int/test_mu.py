from triqs.gf import *
import numpy as np

L = 0.1
w0 = 1.0
beta = 20.0
U = 4.0

q_tau = GfImTime(indices=[0],  statistic='Boson', beta=beta, n_points=4096)
q_iw = make_gf_from_fourier(q_tau)

for w in q_iw.mesh:
    q_iw[w] = 2 * L/w0 * w0**2 / (w.value.imag**2 - w0**2)

q_tau << Fourier(q_iw)
q_tau_mean = np.mean(q_tau.data[:, 0, 0])
print(f"q_tau_mean = {q_tau_mean}")

Q_tau = Block2Gf(['up', 'down'], ['up', 'down'], [[1*q_tau, 1*q_tau], [1*q_tau, 1*q_tau]])
Q_iw = make_gf_from_fourier(Q_tau)

ivn = np.array([x.imag for x in Q_iw["up", "up"].mesh.values()])
zero_freq = np.where(np.abs(ivn) < 1e-10)
q_iw_0 = np.real(Q_iw["up", "up"].data[zero_freq][0,0,0])
print(f"Q_iw(0) = {q_iw_0}")
