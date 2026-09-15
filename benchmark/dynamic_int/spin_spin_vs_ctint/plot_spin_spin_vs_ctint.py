# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.

# Minimal comparison of CTHYB and CTSEG against CTINT for the spin-spin benchmark.
# Columns: full S.S, Jperp only, Sz.Sz only. Rows: G_up(tau), <Sz(tau)Sz(0)>, <Sz(tau)Sz(0)> minus CTINT.
import os
import numpy as np
import matplotlib.pyplot as plt
from h5 import HDFArchive

J = 1.0
U = 4.0
beta = 10.0
data_dir = "/home/andrewhardy/Documents/Data/CTHYB_Data/spin_spin_vs_ctint"
CASES = [((1, 1), r"$\mathbf{S}\cdot\mathbf{S}$"), ((1, 0), r"$J_\perp$ only"), ((0, 1), r"$S_zS_z$ only")]
# series name -> (solver, filename tag); series without files are skipped
SERIES = {"ctint": ("ctint", ""), "ctseg": ("ctseg", ""), "cthyb": ("cthyb", "_lf-True"), "cthyb lf=False": ("cthyb", "_lf-False")}
STYLE = {"ctint": dict(color="green", linestyle="-", linewidth=3.0, label="CTINT"),
         "ctseg": dict(color="blue", linestyle="--", linewidth=2.0, label="CTSEG"),
         "cthyb": dict(color="orange", linestyle=":", linewidth=2.0, label="CTHYB"),
         "cthyb lf=False": dict(color="red", linestyle="-.", linewidth=1.5, label="CTHYB lf=False")}


def load(name, jperp, szsz):
    solver, tag = SERIES[name]
    f = f"{data_dir}/{solver}_J-{J:g}_U-{U:g}_b-{beta:g}_jperp-{jperp:g}_szsz-{szsz:g}{tag}.h5"
    if not os.path.exists(f):
        return None
    with HDFArchive(f, "r") as A:
        return {k: A[k] for k in ("tau_G", "G_up", "tau_SzSz", "SzSz", "average_sign", "pert_order_jperp") if k in A}


fig, axes = plt.subplots(3, len(CASES), figsize=(15, 11), sharex=True)
for col, ((jperp, szsz), title) in enumerate(CASES):
    runs = {s: load(s, jperp, szsz) for s in SERIES}
    ref = runs["ctint"]
    for solver, r in runs.items():
        if r is None:
            continue
        axes[0, col].plot(r["tau_G"], r["G_up"], **STYLE[solver])
        axes[1, col].plot(r["tau_SzSz"], r["SzSz"], **STYLE[solver])
        if ref is not None and solver != "ctint":
            axes[2, col].plot(r["tau_SzSz"], r["SzSz"] - np.interp(r["tau_SzSz"], ref["tau_SzSz"], ref["SzSz"]), **STYLE[solver])
        p = r.get("pert_order_jperp")
        k = (np.arange(len(p)) * p).sum() / p.sum() if p is not None else np.nan
        print(f"jperp={jperp} szsz={szsz} {solver}: sign = {r.get('average_sign', np.nan):.3f}, <k_Jperp> = {k:.3f}, "
              f"SzSz(beta/2) = {np.interp(beta / 2, r['tau_SzSz'], r['SzSz']):.4f}")
    axes[0, col].set_title(title)
    axes[2, col].axhline(0, color="k", linewidth=0.8)
    axes[2, col].set_xlabel(r"$\tau$")

axes[0, 0].set_ylabel(r"$G_\uparrow(\tau)$")
axes[1, 0].set_ylabel(r"$\langle S_z(\tau) S_z(0) \rangle$")
axes[2, 0].set_ylabel(r"$\Delta\langle S_z(\tau) S_z(0) \rangle$ vs CTINT")
axes[0, 0].legend()
fig.suptitle(f"Single-orbital spin-spin, J={J:g}, U={U:g}, beta={beta:g}")
fig.tight_layout()
plt.show()
