# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# CTHYB against exact diagonalization for the two-orbital Kanamori + phonon benchmark.
# Rows: G_a(tau), Re Sigma_a, Im Sigma_a, chi_ab(tau), and the chi residual against ED.
# Columns: the temperatures present on disk.
#
# ED is the reference here and it is exact (the model is defined with one bath site per
# spin-orbital, so there is no bath-discretisation error; only the phonon truncation,
# which run_ed.py quotes and which is ~1e-10 at the default n_ph). So unlike the
# single-orbital benchmarks, a deviation in the residual panel is a CTHYB error, full stop
# -- there is no "which of the two is right?" ambiguity.
#
# Knobs hardcoded below; missing files are skipped.
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive
from triqs.gfs import Gf, BlockGf  # noqa: F401
from triqs.stat.histograms import Histogram  # noqa: F401
# Those imports only register h5 readers for the solver objects the run files
# also carry. Nothing here uses them, but without them every load warns.

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------------- knobs
DATA_DIR = "/home/andrewhardy/Documents/Data/CTHYB_Data/kanamori_phonon"
BETAS = [10.0, 100.0]
U, J, V, EPS_BATH, OMEGA_0 = 2.0, 0.3, 0.7, 0.0, 1.0
G = (0.7, 0.3)         # (0.5, 0.5) is the uniform-coupling control run
MU = "half"            # "half", or the numeric mu of a doped run as it appears in the tag
N_PH = 24
LF = True              # CTHYB Lang-Firsov routing of the file to load
N_CYCLES = 500000
ORB = 0                # which spin-orbital to show for G and Sigma (index into labels)
W_MAX = 15.0
SAVE_AS = None
# -----------------------------------------------------------------------------------------


def tag(beta):
    return (f"beta-{beta}_U-{U}_J-{J}_V-{V}_eb-{EPS_BATH}_w0-{OMEGA_0}"
            f"_g-{G[0]}-{G[1]}_mu-{MU}")


def load(beta):
    ed_path = os.path.join(DATA_DIR, f"ed_{tag(beta)}_nph-{N_PH}.h5")
    cthyb_path = os.path.join(DATA_DIR, f"cthyb_{tag(beta)}_lf-{LF}_nc-{N_CYCLES}.h5")
    out = {}
    for name, path in (("ED", ed_path), ("CTHYB", cthyb_path)):
        if os.path.exists(path):
            with HDFArchive(path, "r") as A:
                out[name] = {k: A[k] for k in A.keys()}
    return out


betas = [b for b in BETAS if load(b)]
if not betas:
    raise SystemExit(f"No files under {DATA_DIR} matching tag '{tag(BETAS[0])}'.\n"
                     "Check DATA_DIR and the knobs at the top of this script.\n"
                     "ED files are produced by:  bash run_kanamori_phonon.sh ed")

STYLE = {"ED": dict(color="k", linestyle="-", linewidth=2.5),
         "CTHYB": dict(color="#e8710a", linestyle="--", linewidth=1.8)}

fig, axes = plt.subplots(5, len(betas), figsize=(5.4 * len(betas), 15), squeeze=False)

for col, beta in enumerate(betas):
    runs = load(beta)
    ed = runs.get("ED")

    for name, r in runs.items():
        st = STYLE[name]

        # --- G_a(tau). ED stores G[a, tau]; CTHYB stores a BlockGf, so pull the same orbital.
        if name == "ED":
            tau = np.asarray(r["tau"])
            g = np.asarray(r["G"])[ORB]
        else:
            gt = r["G_tau"]
            bl, o = list(gt.indices)[0], ORB
            blocks = [b for b, _ in gt]
            bl = blocks[ORB // 2] if len(blocks) > 1 else blocks[0]
            g = gt[bl].data[:, ORB % 2, ORB % 2].real
            tau = np.linspace(0.0, beta, len(g))
        axes[0][col].plot(tau / beta, g, label=name, **st)

        # --- Sigma, both stored on the same positive-Matsubara convention.
        if "Sigma" in r and "w_n" in r:
            w = np.asarray(r["w_n"])
            sig = np.asarray(r["Sigma"])
            sig = sig[ORB] if sig.ndim == 2 else sig
            keep = w <= W_MAX
            axes[1][col].plot(w[keep], sig[keep].real, label=name, **st)
            axes[2][col].plot(w[keep], sig[keep].imag, label=name, **st)

    # --- The correlation function, in the basis CTHYB actually measures.
    #
    # CTHYB does not measure chi_ab. It measures Q_conserved_tau[i,j] = <O_i(tau) O_j(0)>
    # for the density combinations O_i = sum_a c_ia n_a that COMMUTE with h_loc -- for
    # Kanamori that is N_up and N_down, so a 2x2 matrix, not 4x4. The conserved vectors span
    # a subspace of the density space, so chi_ab cannot be recovered from it: the projection
    # is lossy. The comparison therefore goes the other way -- project the exact ED chi DOWN
    # into the same basis, which is exact and costs one einsum.
    #
    # Equal-time convention: Q_conserved_tau is <O_i(tau) O_j(0)> - <O_i O_j>, with the
    # equal-time OPERATOR PRODUCT subtracted (not <O_i><O_j>, so this is not the connected
    # correlator). The Python Solver adds it back only when measure_density_matrix=True, and
    # the run records which happened in 'equal_time_added'. When it was not added, the
    # comparable ED quantity is chi(tau) - chi(0), since chi(0) IS that same product.
    if ed is not None and "CTHYB" in runs and "Q_conserved_tau" in runs["CTHYB"]:
        r = runs["CTHYB"]
        C = np.atleast_2d(np.asarray(r["conserved_vectors"], dtype=float))
        tau_ed = np.asarray(ed["tau"])
        chi_ed = np.einsum("ia,jb,abt->ijt", C, C, np.asarray(ed["chi"]))
        if not r.get("equal_time_added", False):
            chi_ed = chi_ed - chi_ed[..., :1]

        Q = np.asarray(r["Q_conserved_tau"].data).real            # (n_tau, n_cons, n_cons)
        Q = np.moveaxis(Q, 0, -1)                                 # -> (n_cons, n_cons, n_tau)
        tau_Q = np.linspace(0.0, beta, Q.shape[-1])

        n_cons = Q.shape[0]
        for i in range(n_cons):
            for j in range(i, n_cons):
                lbl = f"$Q_{{{i}{j}}}$"
                axes[3][col].plot(tau_Q / beta, Q[i, j], label=f"CTHYB {lbl}",
                                  linestyle="--", linewidth=1.5)
                axes[3][col].plot(tau_ed / beta, chi_ed[i, j], label=f"ED {lbl}",
                                  color="k", linestyle="-", linewidth=1.0, alpha=0.6)
                axes[4][col].plot(tau_Q / beta,
                                  Q[i, j] - np.interp(tau_Q, tau_ed, chi_ed[i, j]), label=lbl)
        ops = r.get("conserved_operators")
        if ops is not None:
            axes[3][col].set_title("conserved: " + ", ".join(str(o) for o in ops), fontsize=7)

    if ed is not None and "CTHYB" in runs:
        trunc = ed.get("ed_truncation", float("nan"))
        axes[4][col].text(0.02, 0.12, f"ED phonon truncation: {float(trunc):.1e}",
                          transform=axes[4][col].transAxes, fontsize=8)

    n_ed = np.mean(ed["density"]) if ed is not None and "density" in ed else float("nan")
    n_qmc = np.mean(runs["CTHYB"]["density"]) if "CTHYB" in runs and "density" in runs["CTHYB"] else float("nan")
    print(f"[beta={beta:g} g={G}] <n> ED {n_ed:.5f} | CTHYB {n_qmc:.5f}"
          + (f" | sign {runs['CTHYB'].get('average_sign', float('nan')):.3f}" if "CTHYB" in runs else ""))

    axes[0][col].set_title(r"$\beta$" + f"={beta:g}, U={U:g}, J={J:g}, "
                           + f"g={G}, " + r"$\omega_0$" + f"={OMEGA_0:g}")
    for row in (1, 2):
        axes[row][col].axhline(0.0, color="k", linewidth=0.5, alpha=0.3)
        axes[row][col].set_xlabel(r"$\omega_n$")
    for row in (3, 4):
        axes[row][col].set_xlabel(r"$\tau/\beta$")

axes[0][0].set_ylabel(rf"$G_{{{ORB}}}(\tau)$")
axes[1][0].set_ylabel(r"$\mathrm{Re}\,\Sigma(i\omega_n)$")
axes[2][0].set_ylabel(r"$\mathrm{Im}\,\Sigma(i\omega_n)$")
axes[3][0].set_ylabel(r"$\chi_{ab}(\tau)$")
axes[4][0].set_ylabel(r"$\Delta\chi$ vs ED")
for row in range(4):
    axes[row][0].legend(fontsize=8)

fig.suptitle("Two-orbital Kanamori + phonon: CTHYB vs exact diagonalization\n"
             "ED is exact here (bath is one site per spin-orbital by construction), "
             "so any deviation is a CTHYB error", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
if SAVE_AS:
    fig.savefig(SAVE_AS, dpi=150, bbox_inches="tight")
    print(f"wrote {SAVE_AS}")
plt.show()
