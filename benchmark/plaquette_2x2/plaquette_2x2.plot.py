# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""Plot plaquette_2x2.py runs, one PDF next to each archive: Sigma_K(iw_n) and G_K(tau) at the
last iteration, and the convergence of Im Sigma_K(iw_0). Y is not shown, being X by symmetry.
G_K(tau) is rebuilt from G_l: the binned G_tau is far noisier under use_norm_as_weight.

    python plaquette_2x2.plot.py plaquette_2x2_DCA_U6.00_beta10.0_mu3.000.h5 [more.h5 ...]
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from h5 import HDFArchive
from triqs.gfs import Gf, MeshImTime

K_SHOWN = ("G", "X", "M")
LABELS = {"G": r"$\Gamma=(0,0)$", "X": r"$X=(\pi,0)$", "M": r"$M=(\pi,\pi)$"}

for path in sys.argv[1:]:
    with HDFArchive(path, "r") as A:
        params, mu = A["params"], A["mu"]
        iters = [A[f"it{i}"] for i in range(A["n_iter"])]
    last = iters[-1]

    mesh = last["Sigma_iw"]["up_G"].mesh
    w = np.array([complex(x).imag for x in mesh.values()])
    pos = w > 0
    g_tau = Gf(mesh=MeshImTime(beta=params["beta"], statistic="Fermion", n_tau=401), target_shape=[1, 1])
    tau = np.array([float(x) for x in g_tau.mesh.values()])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    (ax_im, ax_re), (ax_tau, ax_conv) = axes
    for K in K_SHOWN:
        sig = last["Sigma_iw"][f"up_{K}"].data[pos, 0, 0]
        raw = last["Sigma_iw_raw"][f"up_{K}"].data[pos, 0, 0]
        line, = ax_im.plot(w[pos], sig.imag, label=LABELS[K])
        ax_im.plot(w[pos], raw.imag, ".", color=line.get_color(), alpha=0.3, ms=3)
        ax_re.plot(w[pos], sig.real, color=line.get_color(), label=LABELS[K])
        ax_re.plot(w[pos], raw.real, ".", color=line.get_color(), alpha=0.3, ms=3)
        g_tau.set_from_legendre(last["G_l"][f"up_{K}"])
        ax_tau.plot(tau, g_tau.data[:, 0, 0].real, color=line.get_color(), label=LABELS[K])
        ax_conv.plot([it["Sigma_iw"][f"up_{K}"].data[pos, 0, 0][0].imag for it in iters], "o-",
                     color=line.get_color(), label=LABELS[K])

    for ax in (ax_im, ax_re):
        ax.axvspan(params["fit_min_w"], params["fit_max_w"], color="0.9", zorder=0)
        ax.set_xlim(0, 3 * params["fit_max_w"])
        ax.set_xlabel(r"$\omega_n$")
    ax_im.set_ylabel(r"Im $\Sigma_K(i\omega_n)$")
    ax_re.set_ylabel(r"Re $\Sigma_K(i\omega_n)$")
    ax_im.set_title("lines: used, dots: raw Dyson, grey: tail-fit window", fontsize=9)
    ax_tau.set_xlabel(r"$\tau$")
    ax_tau.set_ylabel(r"$G_K(\tau)$")
    ax_conv.set_xlabel("iteration")
    ax_conv.set_ylabel(r"Im $\Sigma_K(i\omega_0)$")
    ax_conv.set_title(rf"$\langle n \rangle$ = {last['density']:.4f}, sign = {last['average_sign']:.3f}",
                      fontsize=9)
    ax_im.legend()

    fig.suptitle(f"{params['scheme']} 2x2: U = {params['U']}, t = {params['t']}, "
                 rf"$\beta$ = {params['beta']}, $\mu$ = {mu}")
    fig.tight_layout()
    fig.savefig(path.replace(".h5", ".pdf"))
    print(f"Saved {path.replace('.h5', '.pdf')}")
