# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""What the run_ctint.py scripts share: CTINT's auxiliary-spin alpha tensor and the DLR
conversions of its inputs and outputs.

Each vertex contributes (n_a - alpha_a)(n_b - alpha_b), alpha = center +- delta over the two
auxiliary spins. A repulsive coupling needs the two shifts in opposite directions, an
attractive one in the same direction, or vertex weights come out negative, and n - alpha only
has a definite sign for alpha outside [0, 1]. `signed_alpha` therefore sets the direction from
the sign of every h_int term and every D0 channel, with center 1/2 and delta = 1/2 + eta at any
filling. The library's automatic alpha (Solver.find_alpha_from_HF_solver) shifts every D0
channel the same way whatever its sign, and centres delta = 0.1 on the Hartree-Fock density.
"""
import numpy as np
from triqs.gfs import (Gf, MeshDLRImFreq, MeshDLRImTime, fit_gf_dlr, inverse, make_gf_dlr, make_gf_dlr_imfreq,
                       make_gf_from_fourier, make_gf_imtime)

from . import kernels, selfenergy

CENTER = 0.5
DELTA = 0.51


def add_alpha_args(parser):
    parser.add_argument("--alpha", choices=["signed", "library"], default="signed",
                        help="'signed': common/ctint.signed_alpha; 'library': triqs_ctint's Hartree-Fock alpha")
    parser.add_argument("--alpha_delta", type=float, default=DELTA, help="Auxiliary-spin shift delta")
    parser.add_argument("--alpha_center", type=float, default=CENTER, help="Centre of the signed shifts")


def signed_alpha(h_int, d0_channels, center=CENTER, delta=DELTA):
    """alpha tensor, shape (n_terms + n_D0, 2, 2, n_s = 2), in triqs_ctint's layout.

    `d0_channels` is the list of D0(tau) arrays in the library's (block1, block2) order over
    gf_struct, one orbital per block, or empty when there is no D0.
    """
    terms = list(h_int)
    alpha = np.zeros((len(terms) + len(d0_channels), 2, 2, 2))
    for s, sgn in enumerate((1, -1)):
        for l, (_, coeff) in enumerate(terms):
            alpha[l, 0, 0, s] = center - np.sign(coeff) * sgn * delta
            alpha[l, 1, 1, s] = center + sgn * delta
        for d, d0 in enumerate(d0_channels):
            sign = np.sign(d0)
            if not np.all(sign == sign[0]):
                raise ValueError(f"D0 channel {d} changes sign in tau, so no single shift direction fits")
            alpha[len(terms) + d, 0, 0, s] = center - sign[0] * sgn * delta
            alpha[len(terms) + d, 1, 1, s] = center + sgn * delta
    return alpha


def alpha_kwargs(args, h_int, d0_channels):
    """The `S.solve` keywords for the chosen --alpha, and a printable summary."""
    if args.alpha == "library":
        return dict(delta=args.alpha_delta), f"  library alpha (delta {args.alpha_delta:g})"
    alpha = signed_alpha(h_int, d0_channels, args.alpha_center, args.alpha_delta)
    lines = [f"  signed alpha (center {args.alpha_center:g}, delta {args.alpha_delta:g}):"]
    for s in range(2):
        lines.append(f"    s = {s}: " + "  ".join(f"({a[0, 0, s]:+.3f}, {a[1, 1, s]:+.3f})" for a in alpha))
    return dict(alpha=alpha, n_s=2), "\n".join(lines)


def dlr_imfreq_from_tau(data, beta, w_max, eps):
    """A tau-sampled retarded coupling as the DLR-imfreq Gf that CTINT takes."""
    return make_gf_dlr_imfreq(fit_gf_dlr(kernels.as_gf(data, beta), w_max=w_max, eps=eps, symmetrize=True))


def set_g0(S, mesh, mu, delta_iw, n_tau, w_max, eps):
    """Hand every block of `S.G0_iw` the G0^-1 = iw + mu - Delta of common/selfenergy, DLR-fitted."""
    g0_iw = Gf(mesh=mesh, target_shape=(1, 1))
    g0_iw << inverse(selfenergy.g0_inverse_iw(mesh, mu, delta_iw))
    g0_dlr = make_gf_dlr_imfreq(fit_gf_dlr(make_gf_from_fourier(g0_iw, n_tau), w_max=w_max, eps=eps, symmetrize=True))
    for _, g0_block in S.G0_iw:
        g0_block.data[:, 0, 0] = g0_dlr.data[:, 0, 0]


def to_uniform_tau(g_iw, n_tau):
    """CTINT's G(iw) (a DLR or a regular mesh, depending on the version) on `n_tau` points."""
    if isinstance(g_iw.mesh, MeshDLRImFreq):
        return make_gf_imtime(make_gf_dlr(g_iw), n_tau)
    return make_gf_from_fourier(g_iw, n_tau)


def to_regular_imfreq(g_iw, mesh):
    """A possibly-DLR G(iw) on the regular Matsubara `mesh`."""
    if not isinstance(g_iw.mesh, MeshDLRImFreq):
        return g_iw
    out = Gf(mesh=mesh, target_shape=g_iw.target_shape)
    dlr = make_gf_dlr(g_iw)
    for w in mesh:
        out[w] = dlr(w)
    return out


def chi_on_uniform_tau(chi_tau, disconnected, n_tau):
    """`(chi_tau, values)`: CTINT's chiAB_tau on `n_tau` points and its first component. A DLR
    fit holds only the connected part, so `disconnected` (<A><B>) is taken out before it."""
    shift = 0.0
    if isinstance(chi_tau.mesh, MeshDLRImTime):
        shift = disconnected
        chi_connected = chi_tau.copy()
        chi_connected.data[...] -= shift
        chi_tau = make_gf_imtime(make_gf_dlr(chi_connected), n_tau)
    return chi_tau, chi_tau.data.reshape(chi_tau.data.shape[0], -1)[:, 0].real + shift
