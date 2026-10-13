# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""What the run_ctint.py scripts share: CTINT's auxiliary-spin alpha tensor and the DLR
conversions of its inputs and outputs.

Each vertex contributes (n_a - alpha_a)(n_b - alpha_b), alpha = center +- delta over the two
auxiliary spins. A repulsive coupling needs the two shifts in opposite directions, an
attractive one in the same direction, or vertex weights come out negative, and n - alpha only
has a definite sign for alpha outside [0, 1]. Every scheme but 'library' therefore sets the
direction from the sign of every h_int term and every D0 channel. --alpha picks the shifts:

  signed   center 1/2, delta = 1/2 + eta (0.51): alpha outside [0, 1], so sign-definite vertex
           factors at any filling.
  centred  the same directions about the density n with a small delta (0.1). No sign guarantee,
           except for a static repulsive U at particle-hole symmetric half filling, where
           alpha_dn = 1 - alpha_up makes every weight U^k det_up^2.
  near     alpha outside [0, 1] like 'signed', but an attractive channel puts both shifts on the
           side nearest n (1 + eta for n > 1/2, -eta below), the same for both auxiliary spins.
  library  triqs_ctint's Solver.find_alpha_from_HF_solver: delta (0.1) about the Hartree-Fock
           density, every D0 channel shifted the same way whatever its sign.

Since <k> = -<S_int> without a sign problem, the shifts add -sum_l w_l avg_s (n - alpha_a)(n - alpha_b)
to the average order, with w = beta U for an h_int term and beta int_0^beta D0 for a D0 channel
(CTINT's action has no 1/2): about 0.26 w per vertex type for 'signed', delta^2 w for 'centred'.
`extra_order` evaluates it, so an alpha scan can compare the change in the measured order with it.
"""
import numpy as np
from triqs.gfs import (Gf, MeshDLRImFreq, MeshDLRImTime, fit_gf_dlr, inverse, make_gf_dlr, make_gf_dlr_imfreq,
                       make_gf_from_fourier, make_gf_imtime)

from . import kernels, selfenergy

CENTER = 0.5
DEFAULT_DELTA = {"signed": 0.51, "centred": 0.1, "near": 0.01, "library": 0.1}


def add_alpha_args(parser):
    parser.add_argument("--alpha", choices=tuple(DEFAULT_DELTA), default="signed",
                        help="Alpha scheme, see common/ctint.py")
    parser.add_argument("--alpha_delta", type=float, default=None,
                        help="Shift delta (eta for 'near'); default " + ", ".join(f"{k} {v:g}" for k, v in DEFAULT_DELTA.items()))
    parser.add_argument("--alpha_center", type=float, default=None,
                        help="Centre of the 'signed' (default 1/2) and 'centred' (default the density) shifts")
    parser.add_argument("--alpha_tag", action="store_true",
                        help="Put the alpha settings in the output file name, for alpha scans")


def alpha_delta(args):
    return DEFAULT_DELTA[args.alpha] if args.alpha_delta is None else args.alpha_delta


def file_tag(args):
    """The output file-name tag of an --alpha_tag run, e.g. 'alpha-centred-d0.1', else ''."""
    if not args.alpha_tag:
        return ""
    return (f"alpha-{args.alpha}-d{alpha_delta(args):g}"
            + ("" if args.alpha_center is None else f"-c{args.alpha_center:g}"))


def coupling_signs(h_int, d0_channels):
    """The sign of every h_int term, then of every D0 channel, in the alpha tensor's order."""
    signs = [float(np.sign(np.real(coeff))) for _, coeff in h_int]
    for d, d0 in enumerate(d0_channels):
        sign = np.sign(d0)
        if not np.all(sign == sign[0]):
            raise ValueError(f"D0 channel {d} changes sign in tau, so no single shift direction fits")
        signs.append(float(sign[0]))
    return signs


def coupling_weights(h_int, d0_channels, beta):
    """w_l of `extra_order`: beta U per h_int term, beta int_0^beta D0 per D0 channel."""
    weights = [beta * float(np.real(coeff)) for _, coeff in h_int]
    for d0 in d0_channels:
        dtau = beta / (len(d0) - 1)
        weights.append(float(beta * dtau * (np.sum(d0) - 0.5 * (d0[0] + d0[-1]))))
    return weights


def alpha_tensor(signs, scheme, delta, center, density=None):
    """alpha, shape (n_terms + n_D0, 2, 2, n_s = 2), in triqs_ctint's layout, for a non-library
    `scheme`. `density` is only read by 'near', to pick the side of an attractive channel."""
    alpha = np.zeros((len(signs), 2, 2, 2))
    eta = delta
    if scheme == "near":
        center, delta = CENTER, 0.5 + eta
    for s, sgn in enumerate((1, -1)):
        for l, sign in enumerate(signs):
            if scheme == "near" and sign < 0:
                if density is None:
                    raise ValueError("--alpha near needs the density for an attractive channel")
                alpha[l, 0, 0, s] = alpha[l, 1, 1, s] = 1.0 + eta if density > 0.5 else -eta
                continue
            alpha[l, 0, 0, s] = center - sign * sgn * delta
            alpha[l, 1, 1, s] = center + sgn * delta
    return alpha


def extra_order(alpha, weights, density):
    """The average order the shifts add at density n per spin-orbital: -sum_l w_l avg_s (n - alpha_a)(n - alpha_b)."""
    shifted = np.mean((density - alpha[:, 0, 0, :]) * (density - alpha[:, 1, 1, :]), axis=-1)
    return -float(np.dot(weights, shifted))


def alpha_kwargs(args, h_int, d0_channels, beta, density=None):
    """`(kwargs, report, record)`: the `S.solve` keywords for the chosen --alpha, a printable
    summary, and what to store with the run. `density` (per spin-orbital, e.g. the calibrated
    filling) centres 'centred', sides 'near' and predicts the extra order."""
    delta = alpha_delta(args)
    record = dict(scheme=args.alpha, delta=delta)
    if args.alpha == "library":
        return dict(delta=delta), f"  library alpha (delta {delta:g})", record

    center = args.alpha_center
    if center is None:
        center = density if args.alpha == "centred" else CENTER
    if center is None:
        raise ValueError("--alpha centred needs --alpha_center when the density is not known")
    alpha = alpha_tensor(coupling_signs(h_int, d0_channels), args.alpha, delta, center, density)
    record.update(center=center, alpha=alpha)

    lines = [f"  {args.alpha} alpha (center {center:g}, delta {delta:g}):"]
    for s in range(2):
        lines.append(f"    s = {s}: " + "  ".join(f"({a[0, 0, s]:+.3f}, {a[1, 1, s]:+.3f})" for a in alpha))
    if density is not None:
        record["extra_order"] = extra_order(alpha, coupling_weights(h_int, d0_channels, beta), density)
        lines.append(f"    order added by the shifts at n = {density:g}: {record['extra_order']:.1f}")
    return dict(alpha=alpha, n_s=2), "\n".join(lines), record


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
