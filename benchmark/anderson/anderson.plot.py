#!/bin/env python

import os
from h5 import *
from triqs.gfs import *
from triqs.gfs.gf_fnt import rebinning_tau
from triqs.plot.mpl_interface import plt, oplot
from matplotlib.backends.backend_pdf import PdfPages

def setup_fig():
    axes = plt.gca()
    axes.set_ylabel('$G(\\tau)$')
    axes.legend(loc='lower center',prop={'size':10})

spin_names = ("up","dn")

pp = PdfPages('G.pdf')
ed_arch = HDFArchive('anderson.ed.h5','r')
# Optional reference produced by pyed_anderson.py
pyed_arch = HDFArchive('anderson.pyed.h5','r') if os.path.exists('anderson.pyed.h5') else None

for use_blocks, use_qn in ((False,False),(True,False),(False,True),(True,True)):
    file_name = "anderson"
    if use_blocks: file_name += ".block"
    if use_qn: file_name += ".qn"
    file_name += ".h5"
    if not os.path.exists(file_name): continue # (TRIQS 4 h5 raises RuntimeError, not IOError)

    mkind = lambda spin: (spin,0) if use_blocks else ("tot",spin_names.index(spin))

    try:
        arch = HDFArchive(file_name,'r')
        plt.clf()

        name_parts = []
        if use_blocks: name_parts.append('Block')
        if use_qn: name_parts.append('QN')
        name = 'cthyb' + (' (' + ', '.join(name_parts) + ')' if len(name_parts) else '')

        for spin in spin_names:
            bn, i = mkind(spin)
            GF = rebinning_tau(arch['G_tau'][bn],500)
            if use_blocks:
                oplot(GF, name=name + "," + {'up':r"$\uparrow\uparrow$",'dn':r"$\downarrow\downarrow$"}[spin])
            else:
                oplot(GF[i,i], name=name + "," + {'up':r"$\uparrow\uparrow$",'dn':r"$\downarrow\downarrow$"}[spin])
            oplot(ed_arch[spin], name="ED," + {'up':r"$\uparrow\uparrow$",'dn':r"$\downarrow\downarrow$"}[spin])
            if pyed_arch is not None:
                oplot(pyed_arch[spin], name="PYED," + {'up':r"$\uparrow\uparrow$",'dn':r"$\downarrow\downarrow$"}[spin])

        setup_fig()
        pp.savefig(plt.gcf())

    except IOError: pass

pp.close()
