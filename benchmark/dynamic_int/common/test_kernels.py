# Copyright (c) 2025--present, The Simons Foundation
# This file is part of TRIQS/cthyb and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.
"""Check that spin_kernel_Q keeps int Q dtau, K'(0) and its shape in tau/beta fixed as beta changes,
which the beta = 100 runs and every half-filling mu rely on.

    python test_kernels.py
"""
import numpy as np

from kernels import SPIN_KERNEL_BETA_REF, SPIN_KERNEL_POLES, spin_kernel_Q

# Sum rules of the stored beta = 10 kernel the poles were fitted to
INT_Q_REF = 1.523985
KPRIME_0_REF = -0.761992


def sum_rules(beta, n_tau=20001):
    """(int Q dtau, K'(0)), with K'(0) = -int Q dtau / 2 for a kernel symmetric about beta/2."""
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

    # The shape in tau/beta must not change (fixed frequencies would give Q(beta/2)/Q(0) ~ 3e-4 at beta = 100)
    shape = [spin_kernel_Q(b, 2001)[1000] / spin_kernel_Q(b, 2001)[0]
             for b in (SPIN_KERNEL_BETA_REF, 100.0)]
    print(f"Q(beta/2)/Q(0) = {shape[0]:.6f} at beta = 10, {shape[1]:.6f} at beta = 100")
    assert abs(shape[0] - shape[1]) < 1e-3, \
        f"the kernel shape in tau/beta changed: {shape[0]} vs {shape[1]}"

    print(f"OK -- {len(SPIN_KERNEL_POLES)} poles, sum rules and shape invariant under "
          f"tau-rescaling")
