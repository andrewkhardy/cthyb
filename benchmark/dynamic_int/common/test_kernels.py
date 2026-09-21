# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""The one property of the tau-rescaled spin kernel that the beta = 100 runs depend on.

Run directly:  python test_kernels.py     (pure numpy, no MC, under a second)

`spin_kernel_Q` reaches beta = 100 by dividing every pole frequency by beta/beta_ref and
multiplying the amplitude by beta_ref/beta. That combination is chosen to hold K'(0) fixed,
because K'(0) is the static part of the retarded interaction and is what the half-filling
mu closed forms in each model.py depend on. If it drifted, the beta = 100 runs would be
solving a differently-coupled model and every mu would be silently wrong -- hence a test
rather than a comment.
"""
import numpy as np

from kernels import SPIN_KERNEL_BETA_REF, SPIN_KERNEL_POLES, spin_kernel_Q

# From the pole table's provenance (see kernels.py): the values at the reference beta.
INT_Q_REF = 1.523985
KPRIME_0_REF = -0.761992


def sum_rules(beta, n_tau=20001):
    """`(int Q dtau, K'(0))`. K'(0) = -int Q dtau / 2 for a kernel symmetric about beta/2,
    which every single-pole term is, so the two are one check on the amplitude and one on
    the convention relating them."""
    tau = np.linspace(0.0, beta, n_tau)
    integral = float(np.trapezoid(np.asarray(spin_kernel_Q(beta, n_tau)), tau))
    return integral, -0.5 * integral


if __name__ == "__main__":
    ref = sum_rules(SPIN_KERNEL_BETA_REF)
    print(f"beta = {SPIN_KERNEL_BETA_REF:g}:  int Q dtau = {ref[0]:.6f}  K'(0) = {ref[1]:.6f}")
    assert abs(ref[0] - INT_Q_REF) < 1e-4 * abs(INT_Q_REF), \
        f"int Q dtau = {ref[0]} at the reference beta, expected {INT_Q_REF}"
    assert abs(ref[1] - KPRIME_0_REF) < 1e-4 * abs(KPRIME_0_REF), \
        f"K'(0) = {ref[1]} at the reference beta, expected {KPRIME_0_REF}"

    for beta in (25.0, 50.0, 100.0):
        got = sum_rules(beta)
        print(f"beta = {beta:g}: int Q dtau = {got[0]:.6f}  K'(0) = {got[1]:.6f}")
        assert abs(got[0] - ref[0]) < 1e-4 * abs(ref[0]), \
            f"int Q dtau drifted from {ref[0]} to {got[0]} at beta = {beta}"
        assert abs(got[1] - ref[1]) < 1e-4 * abs(ref[1]), \
            f"K'(0) drifted from {ref[1]} to {got[1]} at beta = {beta}"

    # The kernel must also stay long-ranged in tau/beta -- that is the whole reason for
    # rescaling the frequencies rather than keeping them fixed. Fixed frequencies would give
    # Q(beta/2)/Q(0) ~ 3e-4 at beta = 100; rescaling keeps the shape, so the ratio is the
    # same at every beta.
    shape = [spin_kernel_Q(b, 2001)[1000] / spin_kernel_Q(b, 2001)[0]
             for b in (SPIN_KERNEL_BETA_REF, 100.0)]
    print(f"Q(beta/2)/Q(0) = {shape[0]:.6f} at beta = 10, {shape[1]:.6f} at beta = 100")
    assert abs(shape[0] - shape[1]) < 1e-3, \
        f"the kernel shape in tau/beta changed: {shape[0]} vs {shape[1]}"

    print(f"OK -- {len(SPIN_KERNEL_POLES)} poles, sum rules and shape invariant under "
          f"tau-rescaling")
