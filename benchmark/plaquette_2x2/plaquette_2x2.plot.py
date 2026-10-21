# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""Plot plaquette_2x2.py runs, PDFs next to each archive (Y is not shown, being X by symmetry):

  _sigma.pdf         the converged Sigma_K(i omega_n), -Im and Re, of the last iteration: the CRM Sigma fed back
                     (filled markers) and the plain Dyson Sigma G0^-1 - G^-1 of the same G (open markers); dashed, the
                     Hartree level U <n> / 2.
  _G_tau.pdf         G_K(tau) of the last iteration, from its DLR fit G_dlr.
  _convergence.pdf   -Im Sigma_K(i omega_n) of every iteration (light to dark), and per iteration the change of Sigma,
                     the average sign, mu and the impurity and lattice fillings.
  _correlations.pdf  with a final solve: the equal-time spin and charge structure factors S(q), the static
                     susceptibilities chi(q), and <S^z_Q(tau) S^z_Q(0)> / 4 at Q = (pi, pi).

Style: lake.mplstyle next to this file, all text through LaTeX when `latex` is on the PATH (`module load texlive`),
mathtext in Computer Modern otherwise; Matsubara data always as markers.

    python plaquette_2x2.plot.py plaquette_2x2_DCA_U6.00_beta10.0_mu3.000_crm.h5 [more.h5 ...]
"""
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from triqs.gfs import make_gf_imtime  # also registers Gf and BlockGf with h5

USETEX = shutil.which("latex") is not None
STYLE = os.path.join(os.path.dirname(os.path.abspath(globals().get("__file__", "plaquette_2x2.plot.py"))), "lake.mplstyle")
INK, INK_2 = "#0b0b0b", "#52514e"
NOTE = 14
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.style.use(STYLE)
plt.rcParams.update({
    "text.usetex": USETEX, "font.family": "serif", "font.serif": ["Computer Modern Roman" if USETEX else "cmr10"],
    "axes.unicode_minus": USETEX, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "text.color": INK, "lines.markersize": 6, "legend.fontsize": NOTE,
})

K_SHOWN = ("G", "X", "M")
K_TEX = {"G": r"$\Gamma = (0, 0)$", "X": r"$X = (\pi, 0)$", "M": r"$M = (\pi, \pi)$"}
COLOR = dict(zip(K_SHOWN, SERIES))
SITES = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
PANEL = 4.6  # inches per panel, for the fonts of lake.mplstyle


def shade(color, x):
    """color mixed with white, x = 0 (white) to 1 (the color)"""
    return tuple(1 - x * (1 - c) for c in to_rgb(color))


def sigma_K(it, key="Sigma_iw"):
    """(omega_n > 0, {K: Sigma_K(i omega_n)}) of an iteration, spin up"""
    mesh = it[key]["up_G"].mesh
    w = np.array([complex(x).imag for x in mesh.values()])
    return w[w > 0], {K: it[key][f"up_{K}"].data[w > 0, 0, 0] for K in K_SHOWN}


def title(params, mu, final):
    tp = rf" $t' = {params['tp']:g}$," if params.get("tp", 0) else ""
    filling = rf"$n = {params['density']:g}$" if params.get("density") is not None else rf"$\mu = {mu:g}$"
    return rf"$2 \times 2$ {params['scheme']}, $U = {params['U']:g}$,{tp} $\beta = {params['beta']:g}$, {filling}"


def sigma_figure(path, params, head, last):
    w, sig = sigma_K(last)
    _, raw = sigma_K(last, "Sigma_iw_raw")
    fig, axes = plt.subplots(2, len(K_SHOWN), figsize=(PANEL * len(K_SHOWN), 1.8 * PANEL), sharex=True, layout="constrained")
    for c, K in enumerate(K_SHOWN):
        for row, part in enumerate((lambda z: -z.imag, np.real)):
            ax = axes[row, c]
            ax.plot(w, part(sig[K]), color=COLOR[K], marker="o", linewidth=1.2, label="CRM (fed back)")
            ax.plot(w, part(raw[K]), color=INK_2, marker="o", markerfacecolor="none", linestyle="none",
                    label=r"Dyson, $G_0^{-1} - G^{-1}$")
        axes[1, c].axhline(params["U"] * last["density"] / 2, color=INK_2, linewidth=0.8, linestyle="--", zorder=0)
        axes[0, c].set_title(K_TEX[K], fontsize=NOTE + 2)
        axes[1, c].set_xlabel(r"$\omega_n$")
    axes[0, 0].set_ylabel(r"$-\mathrm{Im}\,\Sigma_{\mathbf{K}}(i\omega_n)$")
    axes[1, 0].set_ylabel(r"$\mathrm{Re}\,\Sigma_{\mathbf{K}}(i\omega_n)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(head + rf": converged $\Sigma$ (dashed: $U \langle n \rangle / 2$)")
    fig.savefig(path.replace(".h5", "_sigma.pdf"))
    plt.close(fig)


def g_tau_figure(path, head, last):
    fig, ax = plt.subplots(figsize=(1.4 * PANEL, PANEL), layout="constrained")
    for K in K_SHOWN:
        g = make_gf_imtime(last["G_dlr"][f"up_{K}"], 401)
        tau = np.array([float(t) for t in g.mesh.values()])
        ax.plot(tau, g.data[:, 0, 0].real, color=COLOR[K], label=K_TEX[K])
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$G_{\mathbf{K}}(\tau)$")
    ax.legend(loc="best")
    fig.suptitle(head, fontsize=NOTE + 2)
    fig.savefig(path.replace(".h5", "_G_tau.pdf"))
    plt.close(fig)


def convergence_figure(path, head, iters):
    n = len(iters)
    fig, axes = plt.subplots(2, len(K_SHOWN), figsize=(PANEL * len(K_SHOWN), 1.8 * PANEL), layout="constrained")
    sigmas = [sigma_K(it) for it in iters]
    for c, K in enumerate(K_SHOWN):
        ax = axes[0, c]
        for i, (w, sig) in enumerate(sigmas):
            last = i == n - 1
            ax.plot(w, -sig[K].imag, color=shade(COLOR[K], 0.2 + 0.8 * i / max(n - 1, 1)), marker="o",
                    markersize=6 if last else 3, linewidth=2 if last else 0.8, zorder=2 + i)
        ax.set_title(K_TEX[K], fontsize=NOTE + 2)
        ax.set_xlabel(r"$\omega_n$")
        ax.legend(handles=[Line2D([], [], color=shade(COLOR[K], 0.2), marker="o", markersize=3, linewidth=0.8,
                                  label="iteration 0"),
                           Line2D([], [], color=COLOR[K], marker="o", linewidth=2, label=f"iteration {n - 1}")],
                  loc="best")
    axes[0, 0].set_ylabel(r"$-\mathrm{Im}\,\Sigma_{\mathbf{K}}(i\omega_n)$")

    x = np.arange(n)
    change = [max(np.abs(sigmas[i][1][K] - sigmas[i - 1][1][K]).max() for K in K_SHOWN) for i in range(1, n)]
    axes[1, 0].plot(x[1:], change, color=INK, marker="o", linewidth=1.6)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel(r"$\max\,|\Sigma_{\mathrm{it}} - \Sigma_{\mathrm{it}-1}|$")
    axes[1, 1].plot(x, [it["average_sign"] for it in iters], color=INK, marker="o", linewidth=1.6)
    axes[1, 1].set_ylabel("average sign")
    ax = axes[1, 2]
    if "lattice_density" in iters[0]:
        ax.plot(x, [it["density"] for it in iters], color=SERIES[0], marker="o", linewidth=1.6, label="impurity")
        ax.plot(x, [it["lattice_density"] for it in iters], color=SERIES[1], marker="s", linewidth=1.6, label="lattice")
        ax.set_ylabel(r"$\langle n \rangle$")
        right = ax.twinx()
        right.plot(x, [it["mu"] for it in iters], color=INK_2, marker="D", linestyle="--", linewidth=1.2, label=r"$\mu$")
        right.set_ylabel(r"$\mu$")
        handles = ax.get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0]
        ax.legend(handles=handles, loc="best")
    else:  # earlier runs: the impurity filling only
        ax.plot(x, [it["density"] for it in iters], color=SERIES[0], marker="o", linewidth=1.6)
        ax.set_ylabel(r"$\langle n \rangle$")
    for ax in axes[1]:
        ax.set_xlabel("iteration")
        ax.xaxis.get_major_locator().set_params(integer=True)
    fig.suptitle(head + ": convergence of the loop")
    fig.savefig(path.replace(".h5", "_convergence.pdf"))
    plt.close(fig)


def correlation_figure(path, params, head, final):
    """S(q) = 1/4 sum_RR' e^{iq.(R - R')} C_RR' at q = Gamma, X, M from the equal-time <n_a n_b>, a = 4 s + R, of the
    density matrix, chi(q) the same with C(tau) integrated over tau, and <S^z_Q(tau) S^z_Q(0)> / 4"""
    beta = params["beta"]
    nn = np.asarray(final["nn_sites"]).reshape(2, 4, 2, 4)
    nn_tau = make_gf_imtime(final["nn_dlr_sites"], 2001)
    tau = np.array([float(t) for t in nn_tau.mesh.values()])
    static = np.trapezoid(nn_tau.data.real, tau, axis=0).reshape(2, 4, 2, 4)
    n = np.einsum("arar->r", nn)
    s = np.array([0.5, -0.5])
    qs = np.pi * SITES[[0, 1, 3]]  # Gamma, X, M
    U = np.exp(1j * SITES @ qs.T) / 2

    def by_q(C):
        return np.einsum("rk,rs,sk->k", U.conj(), C, U).real

    parts = {"spin": by_q(np.einsum("a,arbs,b->rs", s, nn, s)), "charge": by_q(nn.sum(axis=(0, 2)) - np.outer(n, n)),
             "chi_spin": by_q(np.einsum("a,arbs,b->rs", s, static, s)),
             "chi_charge": by_q(static.sum(axis=(0, 2)) - beta * np.outer(n, n))}
    ylabels = (r"$S_s(\mathbf{q})$", r"$S_c(\mathbf{q})$", r"$\chi_s(\mathbf{q})$", r"$\chi_c(\mathbf{q})$")
    fig, axes = plt.subplots(1, 5, figsize=(PANEL * 5, PANEL), layout="constrained")
    for ax, (key, values), ylabel in zip(axes, parts.items(), ylabels):
        ax.plot(range(3), values, color=INK, marker="o", markersize=8, linestyle="none")
        ax.set_xticks(range(3), [r"$\Gamma$", r"$X$", r"$M$"])
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.set_xlabel(r"$\mathbf{q}$")
        ax.set_ylabel(ylabel)
    szsz = make_gf_imtime(final["szsz_Q_dlr"], 2001)
    t = np.array([float(x) for x in szsz.mesh.values()])
    y = szsz.data.real.reshape(len(t)) / 4
    axes[4].plot(t, y, color=INK, label=rf"$\chi = {np.trapezoid(y, t):.3g}$, $D = {np.mean([nn[0, r, 1, r] for r in range(4)]):.4f}$")
    axes[4].set_xlabel(r"$\tau$")
    axes[4].set_ylabel(r"$\langle S^z_{\mathbf{Q}}(\tau) S^z_{\mathbf{Q}}(0) \rangle / 4$")
    axes[4].legend(loc="upper center")
    fig.suptitle(head + r": final solve ($\chi$: static AF susceptibility, $D$: double occupancy)")
    fig.savefig(path.replace(".h5", "_correlations.pdf"))
    plt.close(fig)


for path in sys.argv[1:]:
    with HDFArchive(path, "r") as A:
        params, mu = A["params"], A["mu"]
        iters = [A[f"it{i}"] for i in range(A["n_iter"])]
        final = A["final"] if "final" in A else None
    if "G_dlr" not in iters[-1]:
        print(f"Skipping {path}: a run from before the DLR loop")
        continue
    last = iters[-1]
    head = title(params, last.get("mu", mu), final)
    sigma_figure(path, params, head, last)
    g_tau_figure(path, head, last)
    if len(iters) > 1:
        convergence_figure(path, head, iters)
    if final is not None and "nn_dlr_sites" in final:
        correlation_figure(path, params, head, final)
    print(f"Saved the figures of {path}")
