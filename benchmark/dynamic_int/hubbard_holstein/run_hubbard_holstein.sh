#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=ccq
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=23:00:00
#
# Single-orbital Hubbard-Holstein benchmark: CTHYB and CTSEG against CTINT.
# Model and conventions: model.py
#
#   usage:  sbatch run_hubbard_holstein.sh cthyb
#           sbatch run_hubbard_holstein.sh ctseg
#           sbatch run_hubbard_holstein.sh ctint
#
# One job per solver, so each gets its own wall-clock budget. (Formerly forced by
# incompatible module stacks; triqs/multiorbital now carries all three.)
#
# Grid: beta in {10, 100} x n in {0.5, 0.75}, plus the lang_firsov True/False pair for
# CTHYB at the reference point, which is the internal cross-check (uniform g routes every
# vertex analytically, so forcing the stochastic path must reproduce the same physics).
#
# Plot with:  python plot_hubbard_holstein.py

set -euo pipefail
SOLVER="${1:-}"
case "$SOLVER" in
  cthyb|ctseg|ctint) ;;
  *) echo "usage: sbatch $0 cthyb|ctseg|ctint" >&2; exit 2 ;;
esac

# One stack for all three solvers: triqs/multiorbital now carries cthyb, ctseg and ctint.
module load modules/2.5-beta1
module load triqs/multiorbital

NRANKS=96
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/hubbard_holstein

# Two couplings, for two different jobs.
#
# G_MAIN = 0.7 (polaron shift g^2/omega_0^2 = 0.49, i.e. 0.12 U at U = 4) is the physically
# interesting point: a clearly visible retarded effect, still short of the bipolaronic
# regime. CTHYB and CTSEG handle it with sign 1.0.
#
# CTINT used to be restricted to G_WEAK. With triqs_ctint's automatic alpha (Hartree-Fock
# centre, delta = 0.1) its sign collapsed as the polaron shift grew -- measured at beta = 10,
# U = 4, 15k cycles:
#
#     g      g^2/w0^2    sign     <k>
#     0.0    0.00        1.000    6.3     (pure Hubbard: no retarded term at all)
#     0.3    0.09        0.786    6.9
#     0.5    0.25        0.225    9.2
#     0.7    0.49        0.016   13.6
#
# run_ctint.py now uses the signed alpha of common/ctint.py (0.5 +- 0.51, just outside
# [0, 1]), which measured, 16 ranks, short runs, g = 0.7 at the mu below:
#
#     beta   n       sign     <k>
#     10     0.5     1.000    18.8
#     10     0.75    1.000    13.1
#     100    0.5     0.998   186.5
#     100    0.75    0.977   130.6
#
# so CTINT now runs every point CTHYB and CTSEG run, and the three-way check covers the
# physically interesting coupling too. G_WEAK stays as the point where the retarded term is
# a small correction on top of the Hubbard U.
G_MAIN=0.7
G_WEAK=0.3
MODEL_BASE="--U 4.0 --omega_0 1.0 --bath semicircular --out_dir $OUT"
MODEL="$MODEL_BASE --g $G_MAIN"

# ---------------------------------------------------------------------------------------
# Statistics and wall-clock
# ---------------------------------------------------------------------------------------
# n_cycles is per MPI rank, so wall time is the single-rank time.
#
# Measured single-core, beta = 10, U = 4, g = 0.7, omega_0 = 1, semicircular bath:
#   cthyb  average order 6.4, sign 1.00 (every vertex analytic, as uniform g should give)
# That order is ~2.7x the spin-spin benchmark's 2.36, and the determinant cost grows with
# it, so this benchmark is meaningfully slower per cycle than spin_spin at the same beta.
# MAX_TIME is the hard guarantee; each run reports its achieved statistics.
#
# WALL-CLOCK BUDGET -- worst case is (number of `run` calls) x MAX_TIME plus startup and
# the final h5 write. The cthyb path makes 6 calls, so 6 x 1200 s = 120 min: exactly the
# old --time=02:00:00, i.e. no margin at all. Raised to 03:00:00 for the same reason as
# spin_spin. NC_B100 is deliberately NOT reduced here: the overruns were reported for
# spin_spin and vb_dimer, and this benchmark does not save perturbation_order_dyn, so
# there is no measured order to justify a cut. If a beta = 100 call here does run into
# MAX_TIME, halve NC_B100 as the other two scripts now do.
MAX_TIME=1200
case "$SOLVER" in
  cthyb) NC_B10=500000;  NC_B100=50000  ;;
  ctseg) NC_B10=2000000; NC_B100=200000 ;;
  ctint) NC_B10=3000000; NC_B100=300000 ;;
esac

# Every run, at both temperatures and in every solver, uses the one grid of common/grids.py
# (4001 tau points for G, the kernel and every correlator; 1025 frequencies), so the plots
# compare solvers point by point. beta = 100 used to get a finer bosonic grid for K'(0),
# which is Simpson quadrature on it: the relative error at omega_0*beta/2 = 50 is 3e-8 on
# 2001 points and ~2e-9 on 4001, well inside check_half_filling_mu's 1e-5.

# ---------------------------------------------------------------------------------------
# Chemical potentials
# ---------------------------------------------------------------------------------------
# Half filling is exact and needs no value: mu = U/2 - g^2/omega_0^2 = 1.51 here, derived
# from static_shift/half_filling_mu and asserted by every run.
#
# Recalibrated 2026-09-21 with calibrate_mu.py (CTSEG probe, 20k cycles/point, U=4, g=0.7,
# omega_0=1, semicircular bath).
#   python calibrate_mu.py --beta 10  --target_n 0.75
#   python calibrate_mu.py --beta 100 --target_n 0.75
#
# The 2026-09-18 values (3.426927 / 3.412964) predate a calibrate_mu.py fix: it reported an
# interpolated mu while quoting the bisection's achieved n, so the two came from different
# estimators. Both are now the bisection endpoint. Each moved by ~0.011.
#
# Unlike the spin-spin benchmark, n(mu) here is gentle and almost beta-independent, because
# the Holstein coupling shifts the level uniformly rather than opening a low-energy feature.
# The probe reproduces the exact half-filling mu to 1e-4, which is a free check on the
# closed form mu = U/2 - g^2/omega_0^2 = 1.51.
#
# The two values below being bit-identical is NOT a copy-paste slip. The bisection is
# deterministic in mu (same mu_guess, same 0.5 bracketing step, same halving), so both
# temperatures terminate on the same grid point; only the achieved n differs, and the
# deviation changes sign, so the true mu sits just above 3.41625 at beta = 10 and just
# below it at beta = 100. That they land in one cell at all is the beta-independence above.
MU_B10_N075=3.416250    # -> n = 0.749263  (deviation -7.4e-4)
MU_B100_N075=3.416250   # -> n = 0.751271  (deviation +1.3e-3)

run () {  # run <beta> <n_cycles> [extra...]
  local beta="$1"; local ncyc="$2"; shift 2
  echo "=== $SOLVER  beta=$beta  n_cycles=$ncyc  $* ==="
  mpirun -n "$NRANKS" python "run_${SOLVER}.py" $MODEL \
      --beta "$beta" --n_cycles "$ncyc" --n_warmup_cycles $((ncyc / 20)) \
      --max_time "$MAX_TIME" "$@"
}

# ---------------------------------------------------------------- core grid: half filling
# The same list for every solver, so every point has all three.
run 10  "$NC_B10"  --filling 0.5
run 100 "$NC_B100" --filling 0.5
# The weak-coupling point.
MODEL_MAIN="$MODEL"; MODEL="$MODEL_BASE --g $G_WEAK"
run 10 "$NC_B10" --filling 0.5
MODEL="$MODEL_MAIN"

# ------------------------------------------------------------------ core grid: n = 0.75
if [ -n "$MU_B10_N075" ]; then
  run 10 "$NC_B10" --filling 0.75 --mu "$MU_B10_N075"
else
  echo "SKIPPING beta=10 n=0.75: set MU_B10_N075 (see calibrate_mu.py)" >&2
fi
if [ -n "$MU_B100_N075" ]; then
  run 100 "$NC_B100" --filling 0.75 --mu "$MU_B100_N075"
else
  echo "SKIPPING beta=100 n=0.75: set MU_B100_N075 (see calibrate_mu.py)" >&2
fi

# -------------------------------------------------- CTHYB internal cross-check (important)
# With uniform g every vertex is eligible for the analytic Lang-Firsov path, so lang_firsov
# True and False solve the same physics by two entirely different routes. Disagreement here
# is a solver bug, not statistics -- this is the single most informative run in the set.
# The stochastic route has a worse sign, hence the extra cycles.
if [ "$SOLVER" = cthyb ]; then
  run 10 $((NC_B10 * 2)) --filling 0.5 --lang_firsov False
fi

# ---------------------------------------------------------------- extended (uncomment)
#run 100 $((NC_B100 * 2)) --filling 0.5 --lang_firsov False
# Coupling sweep at the reference point, to see the retarded effect grow:
#for G in 0.3 0.5 0.9; do
#  mpirun -n "$NRANKS" python "run_${SOLVER}.py" --U 4.0 --g $G --omega_0 1.0 \
#      --bath semicircular --out_dir $OUT --beta 10 --filling 0.5 \
#      --n_cycles "$NC_B10" --n_warmup_cycles $((NC_B10 / 20)) --max_time "$MAX_TIME"
#done
