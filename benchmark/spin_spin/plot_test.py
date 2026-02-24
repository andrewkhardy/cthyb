
import numpy as np
from triqs.gf import *
from h5 import HDFArchive
from triqs.plot.mpl_interface import oplot, oplotr, oploti, plt

filename = 'spin_spin.out.h5'

with HDFArchive(filename, 'r') as a:
    Delta_tau = a['Delta_tau']
    G_tau = a['G_tau']
    G_iw = a['G_iw']
    G_iw_raw = a['G_iw_raw']
    Sigma_iw = a['Sigma_iw']
    Sigma_iw_raw = a['Sigma_iw_raw']

fig, axes = plt.subplots(3, 3, figsize=(14, 10))

plt.sca(axes[0, 0])
plt.title('Delta_tau')
oplotr(Delta_tau['up'], label='up')
oplotr(Delta_tau['down'], label='down')

plt.sca(axes[0, 2])
plt.title('G_tau')
oplotr(G_tau['up'], label='up')
oplotr(G_tau['down'], label='down')

plt.sca(axes[1, 1])
plt.title('G_iw')
oplot(G_iw['up'], label='up')
oplot(G_iw['down'], label='down')

plt.sca(axes[1, 2])
plt.title('G_iw_raw')
oplot(G_iw_raw['up'], label='up')
oplot(G_iw_raw['down'], label='down')

plt.sca(axes[2, 0])
plt.title('Sigma_iw')
oplot(Sigma_iw['up'], label='up')
oplot(Sigma_iw['down'], label='down')

plt.sca(axes[2, 1])
plt.title('Sigma_iw_raw')
oplot(Sigma_iw_raw['up'], label='up')
oplot(Sigma_iw_raw['down'], label='down')

plt.sca(axes[2, 2])
plt.title('Sigma_iw (zoom)')
oplot(Sigma_iw['up'], label='up')
oplot(Sigma_iw['down'], label='down')
plt.xlim([-15, 15])
plt.ylim([-6, 6])

plt.tight_layout()
plt.show()
