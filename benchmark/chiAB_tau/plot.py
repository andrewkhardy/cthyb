# ----------------------------------------------------------------------

import os
import glob
import numpy as np

# ----------------------------------------------------------------------
from triqs.gfs import *
from h5 import HDFArchive

# ----------------------------------------------------------------------    

from triqs.operators import n
from triqs.operators.util.op_struct import set_operator_structure

from triqs.plot.mpl_interface import oplot, oplotr, oploti, plt     

# ----------------------------------------------------------------------

from pyed.ParameterCollection import ParameterCollection
        
# ----------------------------------------------------------------------
if __name__ == '__main__':

    filename = 'data_cthyb.h5'
    print('--> Loading:', filename)
    with HDFArchive(filename, 'r') as s:
        cthyb = s['p']

    if False:
        filename = '../../test/python/chiAB_tau.ref.h5'
        print('--> Loading:', filename)
        with HDFArchive(filename, 'r') as s:
            chiAB_tau_regr = s['chiAB_tau']
        
    filename = 'data_pyed_h_field_0.0000.h5' # optional reference from calc_pyed.py
    pyed = None
    if os.path.exists(filename):
        print('--> Loading:', filename)
        with HDFArchive(filename, 'r') as s:
            pyed = s['p']
        
    plt.figure(figsize=(3.25*2, 8))
    
    subp = [3, 1, 1]
    plt.subplot(*subp); subp[-1] += 1
    
    #oplotr(chiAB_tau_regr, '-', label=r'cthyb regr $-\langle n_\uparrow(\tau) n_\downarrow \rangle$ (should be bad)', alpha=1.0, lw=1.0, zorder=100)
    oplotr(cthyb.chiAB_tau, '-', label=r'cthyb $\langle n_\uparrow(\tau) n_\downarrow \rangle$', alpha=0.5, lw=0.5)
    if pyed is not None:
        oplotr(pyed.chiAB_tau, label=r'pyed $\langle n_\uparrow(\tau) n_\downarrow \rangle$')

    plt.plot(0, cthyb.exp_val, 'or', alpha=0.25, clip_on=False)
    if pyed is not None:
        plt.plot(0, pyed.exp_val, 'xg', alpha=0.25, clip_on=False)
    
    plt.subplot(*subp); subp[-1] += 1
    tau = np.array([ float(t) for t in cthyb.chiAB_tau.mesh])

    if pyed is not None:
        tau_ref = np.array([ float(t) for t in pyed.chiAB_tau.mesh])
        O_ref = pyed.chiAB_tau.data.copy()
        O_interp = np.interp(tau, tau_ref, O_ref)

        chiAB_tau_diff = cthyb.chiAB_tau.copy()
        chiAB_tau_diff.data[:] -= O_interp
        chiAB_tau_diff.name = r'$\Delta$' + 'chiAB_tau'

        oplotr(chiAB_tau_diff, '-', label=r'cthyb - pyed', alpha=0.75)
    
    plt.subplot(*subp); subp[-1] += 1

    oplotr(cthyb.G_tau, '.', label='cthyb G_tau', alpha=0.2)
    if pyed is not None:
        oplotr(pyed.G_tau, label='pyed G_tau')
    plt.plot(0, -1 + cthyb.n_exp * 0.5, 'ob', alpha=0.25, clip_on=False)
    plt.plot(cthyb.G_tau.mesh.beta, -cthyb.n_exp * 0.5, 'ob', alpha=0.25, clip_on=False)
    
    #oplotr(cthyb.Gl_tau, '-', label='cthyb Gl_tau', alpha=0.75)

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('figure_n_n_by_insertion_cthyb.pdf')
    plt.show()
    
