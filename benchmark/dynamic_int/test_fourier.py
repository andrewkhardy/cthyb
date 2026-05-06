from triqs.gf import *
n_tau = 4096
n_iw = 1024
beta = 10.0
g0 = GfImFreq(indices=[0], beta=beta, n_points=n_iw)
g0 << SemiCircular(2.0)
gt = GfImTime(indices=[0], statistic='Fermion', beta=beta, n_points=n_tau)
gt << Fourier(g0)
print("Fermionic success")
