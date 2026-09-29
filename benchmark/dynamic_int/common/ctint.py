# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
r"""CTINT's auxiliary-spin alpha tensor, shared by every benchmark that runs CTINT.

Each vertex contributes (n_a - alpha_a)(n_b - alpha_b), alpha = center +- delta over the two
auxiliary spins. For a repulsive coupling the two shifts must go in *opposite* directions,
for an attractive one in the *same* direction, or the vertex weights come out negative.

The library's automatic alpha (Solver.find_alpha_from_HF_solver) gets two things wrong for
these benchmarks:

  * It gives every D0 channel the same-direction shift regardless of its sign. That is
    right for an attractive D0 (Hubbard-Holstein, D0 < 0 in every channel) and wrong for a
    repulsive one; the spin-spin D0_ss' = +-spin_kernel/8 has both, so one of the two was
    always shifted the wrong way.
  * It centres the shifts on the Hartree-Fock density with delta = 0.1, i.e. *inside*
    [0, 1]. The equal-time factor n - alpha only has a definite sign when alpha lies
    outside [0, 1], so the shifts belong just below 0 and just above 1: center 1/2,
    delta = 1/2 + eta, at every filling. At beta = 100, n = 0.75, U only, centre 0.75 gave
    sign 0.41 and the library's 0.90 +- 0.1 gave 0.00, against 1.00 for 0.5 +- 0.51.

`signed_alpha` applies the static rule to each h_int term and each D0 channel alike.
"""
import numpy as np

CENTER = 0.5
DELTA = 0.51


def add_alpha_args(parser):
    """The alpha knobs, identical for every benchmark's run_ctint.py."""
    parser.add_argument("--alpha", choices=["signed", "library"], default="signed",
                        help="'signed': explicit alpha tensor, each shift direction set by the sign of "
                             "its coupling (common/ctint.py). 'library': triqs_ctint's automatic "
                             "Hartree-Fock alpha")
    parser.add_argument("--alpha_delta", type=float, default=DELTA,
                        help="Auxiliary-spin shift delta. Larger improves the sign at the cost of a "
                             "higher perturbation order (the library's own default is 0.1)")
    parser.add_argument("--alpha_center", type=float, default=CENTER,
                        help="Centre of the signed alpha shifts. Keep 0.5 at every filling: with "
                             "delta = 0.5 + eta the shifts sit just outside [0, 1]")


def signed_alpha(h_int, d0_channels, center=CENTER, delta=DELTA):
    """alpha tensor, shape (n_terms + n_D0, 2, 2, n_s = 2), in triqs_ctint's layout.

    `d0_channels` is the list of D0(tau) arrays in the library's own order -- (block1,
    block2) over gf_struct, one orbital per block -- or empty when there is no D0.
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
