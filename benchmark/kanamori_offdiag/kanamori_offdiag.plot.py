#!/bin/env python

import os
from h5 import *
from triqs.gfs import *
from triqs.gfs.gf_fnt import rebinning_tau
from triqs.plot.mpl_interface import *
from matplotlib.backends.backend_pdf import PdfPages
from itertools import product

def setup_fig():
    axes = plt.gca()
    axes.set_ylabel('$G(\\tau)$')
    axes.legend(loc='lower center',prop={'size':8}, ncol=1)

spin_names = ("up","dn")
n_orb = 2

pp = PdfPages('G.pdf')
ed_arch = HDFArchive('kanamori_offdiag.ed.h5','r')
# Optional references produced by pyed_kanamori_offdiag.py and pomerol_kanamori_offidag.py
pyed_arch = HDFArchive('kanamori_offdiag.pyed.h5','r') if os.path.exists('kanamori_offdiag.pyed.h5') else None
pomerol_arch = HDFArchive('kanamori_offdiag.pomerol.h5','r') if os.path.exists('kanamori_offdiag.pomerol.h5') else None

for use_qn in (True,False):
    file_name = "kanamori_offdiag"
    if use_qn: file_name += ".qn"
    file_name += ".h5"
    if not os.path.exists(file_name): continue # (TRIQS 4 h5 raises RuntimeError, not IOError)

    try:
        arch = HDFArchive(file_name,'r')

        name = "cthyb (QN)" if use_qn else "cthyb"

        GF_up = rebinning_tau(arch['G_tau']['up'],200)
        GF_dn = rebinning_tau(arch['G_tau']['dn'],200)

        ed_opt = dict(lw=2.0, alpha=1.0)
        cthyb_opt = dict(lw=1.0, alpha=1.0)
        
        for o1, o2 in product(range(n_orb), repeat=2):
            plt.clf()
            plt.title('using_qn = ' + str(use_qn))
            oplot(ed_arch['up'][o1,o2], name=r"ED,$\uparrow%i%i$"%(o1,o2), **ed_opt)
            if pyed_arch is not None:
                oplot(pyed_arch['up'][o1,o2], name=r"PYED,$\uparrow%i%i$"%(o1,o2), **ed_opt)
            if pomerol_arch is not None:
                oplot(pomerol_arch['up']['up'][o1,o2], 'o', name=r"Pomerol,$\uparrow%i%i$"%(o1,o2), **ed_opt)
            oplotr(GF_up[o1,o2], name=name+r",$\uparrow%i%i$"%(o1,o2), **cthyb_opt)
            oploti(GF_up[o1,o2], name=name+r",$\uparrow%i%i$"%(o1,o2), **cthyb_opt)
            setup_fig()
            pp.savefig(plt.gcf())

            plt.clf()
            plt.title('using_qn = ' + str(use_qn))
            oplot(ed_arch['dn'][o1,o2], name=r"ED,$\downarrow%i%i$"%(o1,o2), **ed_opt)
            if pyed_arch is not None:
                oplot(pyed_arch['dn'][o1,o2], name=r"PYED,$\downarrow%i%i$"%(o1,o2), **ed_opt)
            if pomerol_arch is not None:
                oplot(pomerol_arch['dn']['do'][o1,o2], 'o', name=r"Pomerol,$\downarrow%i%i$"%(o1,o2), **ed_opt)
            oplotr(GF_dn[o1,o2], name=name+r",$\downarrow%i%i$"%(o1,o2), **cthyb_opt)
            oploti(GF_dn[o1,o2], name=name+r",$\downarrow%i%i$"%(o1,o2), **cthyb_opt)
            setup_fig()
            pp.savefig(plt.gcf())
            
    except IOError: pass

pp.close()
