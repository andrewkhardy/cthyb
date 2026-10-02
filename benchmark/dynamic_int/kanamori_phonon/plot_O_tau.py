# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# O_tau = <n_B(tau) n_A(0)> from run_cthyb.py --measure_O_tau A B, by random insertion and by the
# exact sweep (--measure_O_tau_min_ins < 0), against ED's chi[B, A]. Top: the correlator; bottom:
# its residual against ED, which is exact up to the phonon truncation. Knobs below; runs cell by
# cell in Jupyter.
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive
from triqs.gfs import Gf  # noqa: F401
# Unused, but it registers the h5 reader for O_tau.

# ---------------------------------------------------------------------------------- knobs
DATA_DIR = "/mnt/home/ahardy/ceph/CTHYB_Data/kanamori_phonon"
TAG = "beta-10.0_U-2.0_J-0.3_V-0.7_eb-0.0_w0-1.0_g-0.5-0.5_mu-half"  # model.Model.tag()
N_PH = 24
SAVE_AS = None  # e.g. "O_tau_vs_ed.pdf"
# -----------------------------------------------------------------------------------------


def load(path):
    with HDFArchive(path, "r") as A:
        return {k: A[k] for k in A.keys()}


def collect():
    """The ED reference and every CTHYB run of TAG that measured O_tau, by file name."""
    ed = load(os.path.join(DATA_DIR, f"ed_{TAG}_nph-{N_PH}.h5"))
    paths = sorted(glob.glob(os.path.join(DATA_DIR, f"cthyb_{TAG}_lf-*_Otau-*.h5")))
    return ed, {os.path.basename(p): load(p) for p in paths}


def estimator(run):
    return "sweep" if run["O_tau_min_ins"] < 0 else f"insertion ({run['O_tau_min_ins']})"


def plot(ed, runs):
    beta = ed["params"]["beta"]
    fig, (ax, ax_res) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    pairs_drawn = set()
    for name, run in runs.items():
        a, b = (int(x) for x in run["O_tau_pair"])
        tau = np.array([float(t) for t in run["O_tau"].mesh])
        O_tau = run["O_tau"].data.real
        chi_ed = np.interp(tau, ed["tau"], ed["chi"][b, a])
        if (a, b) not in pairs_drawn:
            pairs_drawn.add((a, b))
            ax.plot(ed["tau"] / beta, ed["chi"][b, a], color="k", linewidth=2.5,
                    label=f"ED  <n_{b}(tau) n_{a}(0)>")
        label = f"{estimator(run)}, {run['solve_seconds']:.0f} s"
        ax.plot(tau / beta, O_tau, linestyle="--", label=label)
        ax_res.plot(tau / beta, O_tau - chi_ed, label=label)
        print(f"{name}\n  {estimator(run)}: max |O_tau - ED| = {np.abs(O_tau - chi_ed).max():.2e}, "
              f"solve {run['solve_seconds']:.0f} s, n_cycles {run['n_cycles']}, sign {run['average_sign']:.4f}")
    ax.set_ylabel(r"$\langle n_B(\tau)\, n_A(0)\rangle$")
    ax_res.set_ylabel("CTHYB - ED")
    ax_res.set_xlabel(r"$\tau/\beta$")
    ax_res.axhline(0.0, color="k", linewidth=0.8)
    ax.legend(fontsize=8)
    ax_res.legend(fontsize=8)
    fig.tight_layout()
    return fig


ed, runs = collect()
fig = plot(ed, runs)
if SAVE_AS:
    fig.savefig(SAVE_AS)
plt.show()
