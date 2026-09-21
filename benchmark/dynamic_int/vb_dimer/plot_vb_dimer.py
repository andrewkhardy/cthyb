# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# Two-patch valence-bond dimer with a retarded real-space spin-spin interaction.
# Rows: G(tau), Re Sigma, Im Sigma, chi^zz(tau), and the chi residual against the reference.
# Columns: the temperatures present on disk.
#
# WHICH REFERENCE APPLIES DEPENDS ON THE COUPLING, and the distinction is physical, not a
# convenience:
#
#   POINT = "ed"   (J_intra = -J_inter)  ED exists and is exact, so it is the reference and
#                                        any CTHYB deviation is a CTHYB error.
#   POINT = "dca"  (J_intra = 0)         NO ED can exist: integrating out harmonic bosons
#                                        only ever gives -(psd) x |Q|, and -J is indefinite
#                                        here. The reference is then CTHYB's own
#                                        lang_firsov=True vs False pair -- two different
#                                        routes to the same physics. Agreement is evidence,
#                                        not proof.
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
DATA_DIR = "/home/andrewhardy/Documents/Data/CTHYB_Data/vb_dimer"
POINT = "ed"           # "ed" (J_intra=-J_inter, exact ED) or "dca" (J_intra=0, no ED)
BETAS = [10.0, 100.0]
T, TP, U, OMEGA_0 = 0.25, 0.0, 2.0, 1.0
N_PH = 3
N_CYCLES = 500000
ORB = 0
W_MAX = 15.0
SAVE_AS = None
# -----------------------------------------------------------------------------------------

if POINT == "ed":
    J_INTRA, J_INTER, BATH, V = -0.5, 0.5, "discrete", 0.5
else:
    J_INTRA, J_INTER, BATH, V = 0.0, 0.5, "dca", 0.5


def find(prefix, beta, *extra):
    """Locate a result file by prefix, matching on every piece of the model tag that
    distinguishes the two coupling points. Matching on beta alone is not enough: the J_ED
    and J_DCA runs sit in the same directory, and 'Jintra--0.5' sorts before 'Jintra-0.0',
    so a loose match silently returns the wrong point's file."""
    if not os.path.isdir(DATA_DIR):
        return None
    bath = f"bath-V-{V}" if BATH == "discrete" else "bath-dca"
    required = (f"beta-{beta}", f"Jintra-{J_INTRA}", f"Jinter-{J_INTER}", bath) + extra
    cands = [f for f in sorted(os.listdir(DATA_DIR))
             if f.startswith(prefix) and all(s in f for s in required)]
    return os.path.join(DATA_DIR, cands[0]) if cands else None


def load(path):
    if path is None or not os.path.exists(path):
        return None
    with HDFArchive(path, "r") as A:
        return {k: A[k] for k in A.keys()}


def runs_for(beta):
    out = {}
    if POINT == "ed":
        r = load(find("ed_", beta))
        if r is not None:
            out["ED (exact)"] = r
    for label, tag in (("CTHYB lf=True", "lf-True"), ("CTHYB lf=False", "lf-False")):
        r = load(find("cthyb_", beta, tag))
        if r is not None:
            out[label] = r
    return out


betas = [b for b in BETAS if runs_for(b)]
if not betas:
    raise SystemExit(f"No files under {DATA_DIR} for POINT={POINT!r}.\n"
                     "ED files come from:  sbatch run_vb_dimer.sh ed\n"
                     "(there is no ED for POINT='dca' -- see the header)")

STYLE = {"ED (exact)": dict(color="k", linestyle="-", linewidth=2.5),
         "CTHYB lf=True": dict(color="#e8710a", linestyle="--", linewidth=1.8),
         "CTHYB lf=False": dict(color="#c1121f", linestyle="-.", linewidth=1.5)}

fig, axes = plt.subplots(5, len(betas), figsize=(5.4 * len(betas), 15), squeeze=False)

for col, beta in enumerate(betas):
    runs = runs_for(beta)
    ref_name = "ED (exact)" if "ED (exact)" in runs else "CTHYB lf=True"
    ref = runs.get(ref_name)

    for name, r in runs.items():
        st = STYLE.get(name, dict())

        # Both solvers write G[a, tau] against a shared `tau`, so there is one layout.
        if "G" in r:
            axes[0][col].plot(np.asarray(r["tau"]) / beta, np.asarray(r["G"])[ORB],
                              label=name, **st)

        if "Sigma" in r and "w_n" in r:
            w = np.asarray(r["w_n"])
            sig = np.asarray(r["Sigma"])
            sig = sig[ORB] if sig.ndim == 2 else sig
            keep = w <= W_MAX
            axes[1][col].plot(w[keep], sig[keep].real, label=name, **st)
            axes[2][col].plot(w[keep], sig[keep].imag, label=name, **st)

        chi = r.get("chi_zz")
        if chi is not None:
            chi = np.asarray(chi)
            axes[3][col].plot(np.asarray(r["tau"]) / beta, chi[0, 0], label=f"{name} " + r"$\chi^{zz}_{00}$", **st)
            axes[3][col].plot(np.asarray(r["tau"]) / beta, chi[0, 1],
                              color=st.get("color", "k"), linestyle=":", linewidth=1.2,
                              label=f"{name} " + r"$\chi^{zz}_{01}$")
        elif "corr" in r:
            axes[3][col].plot(np.asarray(r["tau_corr"]) / beta, np.asarray(r["corr"]),
                              label=name, **st)

        if ref is not None and name != ref_name:
            a = r.get("corr")
            b = ref.get("corr")
            if a is not None and b is not None:
                grid = np.linspace(0.0, beta, 61)
                resid = (np.interp(grid, r["tau_corr"], a) - np.interp(grid, ref["tau_corr"], b))
                axes[4][col].plot(grid / beta, resid, label=f"{name} - {ref_name}",
                                  marker="o", markersize=2.5, **st)

        print(f"[beta={beta:g} {POINT}] {name:16s} "
              + (f"sign={r['average_sign']:.3f}  " if "average_sign" in r else "")
              + (f"<n>={np.mean(r['density']):.4f}" if "density" in r else ""))

    if "ED (exact)" in runs:
        trunc = runs["ED (exact)"].get("ed_truncation", float("nan"))
        axes[4][col].text(0.02, 0.06, f"ED phonon truncation: {float(trunc):.1e}",
                          transform=axes[4][col].transAxes, fontsize=8)
    elif POINT == "dca":
        axes[4][col].text(0.02, 0.5,
                          "No ED at this coupling:\n-J is not positive semidefinite,\n"
                          "so no real-boson Hamiltonian exists.\n"
                          "Reference is the lf True/False pair.",
                          transform=axes[4][col].transAxes, fontsize=8, va="center")
    else:
        # -J IS psd here, so an ED exists -- the file just is not there yet.
        axes[4][col].text(0.02, 0.5, "ED reference not found.\nRun:  sbatch run_vb_dimer.sh ed",
                          transform=axes[4][col].transAxes, fontsize=8, va="center")

    axes[0][col].set_title(r"$\beta$" + f"={beta:g}, U={U:g}, "
                           + r"$J_{\rm intra}$" + f"={J_INTRA:g}, "
                           + r"$J_{\rm inter}$" + f"={J_INTER:g}, bath={BATH}")
    axes[4][col].axhline(0.0, color="k", linewidth=0.8)
    for row in (1, 2):
        axes[row][col].axhline(0.0, color="k", linewidth=0.5, alpha=0.3)
        axes[row][col].set_xlabel(r"$\omega_n$")
    for row in (3, 4):
        axes[row][col].set_xlabel(r"$\tau/\beta$")

axes[0][0].set_ylabel(rf"$G_{{{ORB}}}(\tau)$")
axes[1][0].set_ylabel(r"$\mathrm{Re}\,\Sigma(i\omega_n)$")
axes[2][0].set_ylabel(r"$\mathrm{Im}\,\Sigma(i\omega_n)$")
axes[3][0].set_ylabel(r"$\chi^{zz}_{ij}(\tau)$")
axes[4][0].set_ylabel(r"$\Delta\chi$ vs reference")
for row in range(4):
    axes[row][0].legend(fontsize=7)

title = ("Valence-bond dimer, retarded real-space "
         + r"$\mathbf{S}\cdot\mathbf{S}$" + "\n"
         + ("ED is exact at this coupling, so deviations are CTHYB errors"
            if POINT == "ed" else
            "No ED exists at this coupling (-J indefinite); reference is lf True vs False"))
fig.suptitle(title, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
if SAVE_AS:
    fig.savefig(SAVE_AS, dpi=150, bbox_inches="tight")
    print(f"wrote {SAVE_AS}")
plt.show()
