# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
#
# CTHYB against exact diagonalization for the two-orbital Kanamori + phonon benchmark.
# Rows: G_a(tau), Re Sigma_a, Im Sigma_a, chi_ab(tau), and the chi residual against ED.
# Columns: the temperatures present on disk. One figure per filling in FILLINGS.
#
# ED is the reference here and it is exact (the model is defined with one bath site per
# spin-orbital, so there is no bath-discretisation error; only the phonon truncation,
# which run_ed.py quotes and which is ~1e-10 at the default n_ph). So unlike the
# single-orbital benchmarks, a deviation in the residual panel is a CTHYB error, full stop
# -- there is no "which of the two is right?" ambiguity.
#
# Knobs hardcoded below; missing files are skipped.
import glob
import os
import re
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
DATA_DIR = "/mnt/home/ahardy/ceph/CTHYB_Data/kanamori_phonon"  # on the cluster
BETAS = [10.0, 100.0]
FILLINGS = [0.5, 0.75]  # one figure each: 0.5 = half filling, 0.75 = the doped runs (see collect)
U, J, V, EPS_BATH, OMEGA_0 = 2.0, 0.3, 0.7, 0.0, 1.0
G = (0.7, 0.3)         # (0.5, 0.5) is the uniform-coupling control run
N_PH = 24
LF = True              # CTHYB Lang-Firsov routing of the file to load
ORB = 0                # which spin-orbital to show for G and Sigma (index into labels)
W_MAX = 15.0
SAVE_AS = None         # e.g. "kanamori_n{filling:g}.pdf"
# -----------------------------------------------------------------------------------------

STYLE = {"ED": dict(color="k", linestyle="-", linewidth=2.5),
         "CTHYB": dict(color="#e8710a", linestyle="--", linewidth=1.8)}
# Sigma(iw): ED filled circles, CTHYB open squares drawn around them
MARKER = {"ED": dict(marker="o", markersize=3.0, markerfacecolor="k"),
          "CTHYB": dict(marker="s", markersize=4.5, markerfacecolor="none")}


def matsubara_style(name):
    """Every Matsubara point marked, joined by a thin dotted line: the data are discrete, and
    a solid curve would hide where the points actually are."""
    return dict(color=STYLE[name]["color"], linestyle=":", linewidth=0.8, markeredgewidth=0.9,
                **MARKER[name])


def base_tag(beta):
    return f"beta-{beta}_U-{U}_J-{J}_V-{V}_eb-{EPS_BATH}_w0-{OMEGA_0}_g-{G[0]}-{G[1]}"


def load(path):
    with HDFArchive(path, "r") as A:
        return {k: A[k] for k in A.keys()}


def collect(beta):
    """`{filling: (mu, {"ED": run, "CTHYB": run})}` for everything on disk at this beta.

    Files are grouped by the mu in their name, so ED and CTHYB in one column always solved
    the same Hamiltonian. Each group goes to the filling in FILLINGS nearest its <n> -- ED's
    when there is one, since it is exact. Within a group the newest file wins: the n_cycles
    in a CTHYB name differs between beta = 10 and 100, and reruns can sit beside old files.
    """
    patterns = {"ED": f"ed_{base_tag(beta)}_mu-*_nph-{N_PH}.h5",
                "CTHYB": f"cthyb_{base_tag(beta)}_mu-*_lf-{LF}_nc-*.h5"}
    groups = {}
    for name, pattern in patterns.items():
        for path in glob.glob(os.path.join(DATA_DIR, pattern)):
            if "_seed-" in path:   # independent chains for diagnose_residual.py, not a result
                continue
            mu = re.search(r"_mu-([^_]+)_", os.path.basename(path)).group(1)
            groups.setdefault(mu, {}).setdefault(name, []).append(path)

    out = {}
    for mu, files in groups.items():
        paths = {name: max(found, key=os.path.getmtime) for name, found in files.items()}
        runs = {name: load(path) for name, path in paths.items()}
        n_mean = float(np.mean(runs.get("ED", runs.get("CTHYB"))["density"]))
        filling = min(FILLINGS, key=lambda f: abs(f - n_mean))
        if abs(filling - n_mean) > 0.1:
            print(f"[beta={beta:g}] mu={mu}: <n> = {n_mean:.3f} is near none of {FILLINGS} -- skipped")
            continue
        newest = max(os.path.getmtime(path) for path in paths.values())
        if filling in out:
            keep, drop = (mu, out[filling][0]) if newest > out[filling][2] else (out[filling][0], mu)
            print(f"[beta={beta:g} n={filling:g}] mu={drop} also lands here; keeping the newer mu={keep}")
            if keep != mu:
                continue
        out[filling] = (mu, runs, newest)
        for name, path in paths.items():
            print(f"[beta={beta:g} n={filling:g}] {name}: {os.path.basename(path)}")
    return {filling: (mu, runs) for filling, (mu, runs, _) in out.items()}


def make_figure(filling, columns):
    """One 5 x n_beta grid for this filling; `columns` is [(beta, mu, runs)]."""
    fig, axes = plt.subplots(5, len(columns), figsize=(5.4 * len(columns), 15), squeeze=False)

    for col, (beta, mu, runs) in enumerate(columns):
        ed = runs.get("ED")

        for name, r in runs.items():
            st = STYLE[name]

            # --- G_a(tau). ED stores G[a, tau]; CTHYB stores a BlockGf, so pull the same orbital.
            if name == "ED":
                tau = np.asarray(r["tau"])
                g = np.asarray(r["G"])[ORB]
            else:
                gt = r["G_tau"]
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
                axes[1][col].plot(w[keep], sig[keep].real, label=name, **matsubara_style(name))
                axes[2][col].plot(w[keep], sig[keep].imag, label=name, **matsubara_style(name))

        # --- The correlation function, in the basis CTHYB actually measures.
        #
        # CTHYB does not measure chi_ab. It measures Q_conserved_tau[i,j] = <O_i(tau) O_j(0)>
        # for the density combinations O_i = sum_a c_ia n_a that COMMUTE with h_loc -- for
        # Kanamori that is N_up and N_down, so a 2x2 matrix, not 4x4. The conserved vectors
        # span a subspace of the density space, so chi_ab cannot be recovered from it: the
        # projection is lossy. The comparison therefore goes the other way -- project the
        # exact ED chi DOWN into the same basis, which is exact and costs one einsum.
        #
        # Equal-time convention: Q_conserved_tau is <O_i(tau) O_j(0)> - <O_i O_j>, with the
        # equal-time OPERATOR PRODUCT subtracted (not <O_i><O_j>, so this is not the connected
        # correlator). The Python Solver adds it back only when measure_density_matrix=True,
        # and the run records which happened in 'equal_time_added'. When it was not added,
        # the comparable ED quantity is chi(tau) - chi(0), since chi(0) IS that same product.
        #
        # Say why the residual panel is empty rather than leave a blank box.
        why_empty = None
        if ed is None:
            why_empty = "no ED reference at this point\nRun:  bash run_kanamori_phonon.sh ed"
        elif "CTHYB" not in runs:
            why_empty = "no CTHYB run at this point"
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
            # Point by point: CTHYB and ED both write on the grid of common/grids.py. A file
            # from before that is skipped, not resampled.
            same_grid = len(tau_ed) == len(tau_Q) and np.allclose(tau_ed, tau_Q)
            if not same_grid:
                print(f"[beta={beta:g} n={filling:g}] CTHYB Q has {len(tau_Q)} tau points against "
                      f"ED's {len(tau_ed)} -- residual skipped, rerun on the shared grid")
                why_empty = (f"CTHYB Q on {len(tau_Q)} tau points, ED on {len(tau_ed)}:\n"
                             "rerun both on the shared grid")

            n_cons = Q.shape[0]
            for i in range(n_cons):
                for j in range(i, n_cons):
                    lbl = f"$Q_{{{i}{j}}}$"
                    axes[3][col].plot(tau_Q / beta, Q[i, j], label=f"CTHYB {lbl}",
                                      linestyle="--", linewidth=1.5)
                    axes[3][col].plot(tau_ed / beta, chi_ed[i, j], label=f"ED {lbl}",
                                      color="k", linestyle="-", linewidth=1.0, alpha=0.6)
                    if same_grid:
                        axes[4][col].plot(tau_Q / beta, Q[i, j] - chi_ed[i, j], label=lbl)
            ops = r.get("conserved_operators")
            if ops is not None:
                axes[3][col].set_title("conserved: " + ", ".join(str(o) for o in ops), fontsize=7)

        if why_empty:
            axes[4][col].text(0.5, 0.5, why_empty, ha="center", va="center", fontsize=9,
                              color="#c1121f", transform=axes[4][col].transAxes)
        if ed is not None and "CTHYB" in runs:
            trunc = ed.get("ed_truncation", float("nan"))
            axes[4][col].text(0.02, 0.12, f"ED phonon truncation: {float(trunc):.1e}",
                              transform=axes[4][col].transAxes, fontsize=8)

        n_ed = np.mean(ed["density"]) if ed is not None and "density" in ed else float("nan")
        n_qmc = np.mean(runs["CTHYB"]["density"]) if "CTHYB" in runs and "density" in runs["CTHYB"] else float("nan")
        print(f"[beta={beta:g} n={filling:g} g={G} mu={mu}] <n> ED {n_ed:.5f} | CTHYB {n_qmc:.5f}"
              + (f" | sign {runs['CTHYB'].get('average_sign', float('nan')):.3f}" if "CTHYB" in runs else ""))
        # Equilibrium, as run_cthyb.py checks it. At beta = 100 Sigma is only worth reading
        # when the two occupation estimators agree to ~1e-3 and warmup >> tau_auto.
        if "CTHYB" in runs and "orbital_occupations" in runs["CTHYB"]:
            r = runs["CTHYB"]
            gap = np.abs(np.asarray(r["orbital_occupations"]) - np.asarray(r["density"])).max()
            tau = (f"tau_auto {r['auto_corr_time']:.0f} cycles vs warmup {r['n_warmup_cycles']} | "
                   if "auto_corr_time" in r else "")
            print(f"    CTHYB {tau}max |n_rho - n_G| {gap:.4f}")

        axes[0][col].set_title(r"$\beta$" + f"={beta:g}, U={U:g}, J={J:g}, "
                               + f"g={G}, " + r"$\omega_0$" + f"={OMEGA_0:g}\n"
                               + r"$\mu$" + f"={mu}, n={filling:g}")
        axes[4][col].axhline(0.0, color="k", linewidth=0.8)
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
    for row in range(5):
        # Only where something was actually drawn, else matplotlib warns about an empty legend.
        if axes[row][0].get_legend_handles_labels()[0]:
            axes[row][0].legend(fontsize=8)

    fig.suptitle(f"Two-orbital Kanamori + phonon, n={filling:g}: CTHYB vs exact diagonalization\n"
                 "ED is exact here (bath is one site per spin-orbital by construction), "
                 "so any deviation is a CTHYB error", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


by_beta = {beta: collect(beta) for beta in BETAS}
drawn = 0
for filling in FILLINGS:
    columns = [(beta, *by_beta[beta][filling]) for beta in BETAS if filling in by_beta[beta]]
    if not columns:
        print(f"[n={filling:g}] no files under {DATA_DIR} for g={G} -- skipped")
        continue
    fig = make_figure(filling, columns)
    drawn += 1
    if SAVE_AS:
        name = SAVE_AS.format(filling=filling)
        fig.savefig(name, dpi=150, bbox_inches="tight")
        print(f"wrote {name}")
if not drawn:
    raise SystemExit(f"No files under {DATA_DIR} for g={G}.\n"
                     "Check DATA_DIR and the knobs at the top of this script.\n"
                     "ED files are produced by:  bash run_kanamori_phonon.sh ed")
plt.show()
