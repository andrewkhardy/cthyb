# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""Minimal comparison of CTHYB and CTSEG against CTINT for the spin-spin benchmark.

Columns: full S.S, Jperp only, Sz.Sz only. Rows: G_up(tau), <Sz(tau)Sz(0)>, and
<Sz(tau)Sz(0)> minus CTINT. Also prints a summary table.

    python plot_spin_spin_vs_ctint.py --J 1.0 --out spin_spin_vs_ctint_J-1.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from h5 import HDFArchive

CASES = [((1, 1), r"$\mathbf{S}\cdot\mathbf{S}$", "full S.S"), ((1, 0), r"$J_\perp$ only", "Jperp only"),
         ((0, 1), r"$S_zS_z$ only", "SzSz only")]
STYLE = {"ctint": dict(color="green", linestyle="-", linewidth=3.0, label="CTINT"),
         "ctseg": dict(color="blue", linestyle="--", linewidth=2.0, label="CTSEG"),
         "cthyb": dict(color="orange", linestyle=":", linewidth=2.0, label="CTHYB")}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--J", type=float, default=1.0)
parser.add_argument("--U", type=float, default=4.0)
parser.add_argument("--beta", type=float, default=10.0)
parser.add_argument("--data_dir", default="/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin_vs_ctint")
parser.add_argument("--cthyb_tag", default="_lf-True", help="Filename tag of the CTHYB runs to plot")
parser.add_argument("--out", default="spin_spin_vs_ctint.png")
args = parser.parse_args()


def load(solver, jperp, szsz):
    tag = args.cthyb_tag if solver == "cthyb" else ""
    f = os.path.join(args.data_dir, f"{solver}_J-{args.J:g}_U-{args.U:g}_b-{args.beta:g}_jperp-{jperp:g}_szsz-{szsz:g}{tag}.h5")
    if not os.path.exists(f):
        return None
    with HDFArchive(f, "r") as A:
        return {k: A[k] for k in ("tau_G", "G_up", "tau_SzSz", "SzSz", "average_sign", "pert_order_jperp") if k in A}


def mean_order(p):
    return (np.arange(len(p)) * p).sum() / p.sum()


fig, axes = plt.subplots(3, len(CASES), figsize=(5 * len(CASES), 11), sharex=True)
print(f"{'case':<14}{'solver':<8}{'sign':>8}{'<k_Jperp>':>11}{'SzSz(b/2)':>11}{'max|dSzSz|':>12}{'max|dG|':>10}   (d = minus CTINT)")
for col, ((jperp, szsz), title, label) in enumerate(CASES):
    runs = {s: load(s, jperp, szsz) for s in ("ctint", "ctseg", "cthyb")}
    ref = runs["ctint"]
    for solver, r in runs.items():
        if r is None:
            continue
        axes[0, col].plot(r["tau_G"], r["G_up"], **STYLE[solver])
        axes[1, col].plot(r["tau_SzSz"], r["SzSz"], **STYLE[solver])
        dG = dSzSz = np.nan
        if ref is not None and solver != "ctint":
            dSzSz_tau = r["SzSz"] - np.interp(r["tau_SzSz"], ref["tau_SzSz"], ref["SzSz"])
            axes[2, col].plot(r["tau_SzSz"], dSzSz_tau, **STYLE[solver])
            dSzSz = np.abs(dSzSz_tau).max()
            dG = np.abs(r["G_up"] - np.interp(r["tau_G"], ref["tau_G"], ref["G_up"])).max()
        k = mean_order(r["pert_order_jperp"]) if "pert_order_jperp" in r else np.nan
        sign = r.get("average_sign", np.nan)
        print(f"{label:<14}{solver:<8}{sign:>8.3f}{k:>11.3f}{np.interp(args.beta / 2, r['tau_SzSz'], r['SzSz']):>11.4f}"
              f"{dSzSz:>12.4f}{dG:>10.4f}")
    axes[0, col].set_title(title)
    axes[2, col].axhline(0, color="k", linewidth=0.8)
    axes[2, col].set_xlabel(r"$\tau$")

axes[0, 0].set_ylabel(r"$G_\uparrow(\tau)$")
axes[1, 0].set_ylabel(r"$\langle S_z(\tau) S_z(0) \rangle$")
axes[2, 0].set_ylabel(r"$\Delta\langle S_z(\tau) S_z(0) \rangle$ vs CTINT")
axes[0, 0].legend()
fig.suptitle(f"Single-orbital spin-spin, J={args.J:g}, U={args.U:g}, beta={args.beta:g}")
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"Figure saved to {args.out}")
