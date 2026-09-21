#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=ccq
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#
# Single-orbital Hubbard-Holstein benchmark: CTHYB and CTSEG against CTINT.
# Model and conventions: model.py
#
#   usage:  sbatch run_hubbard_holstein.sh cthyb
#           sbatch run_hubbard_holstein.sh ctseg
#           sbatch run_hubbard_holstein.sh ctint
#
# One job per solver: the three need mutually incompatible module stacks.
#
# Grid: beta in {10, 100} x n in {0.5, 0.75}, plus the lang_firsov True/False pair for
# CTHYB at the reference point, which is the internal cross-check (uniform g routes every
# vertex analytically, so forcing the stochastic path must reproduce the same physics).
#
# Plot with:  python plot_hubbard_holstein.py

set -euo pipefail
SOLVER="${1:-}"
case "$SOLVER" in
  cthyb) module load modules/2.5-beta1; module load triqs/multiorbital ;;
  ctseg) module load modules/2.5-beta1; module load triqs/unstable ;;
  ctint) module purge
         module load modules/2.4 gcc flexiblas openmpi cmake ccache gmp fftw nfft hdf5/mpi \
                     boost python/3.12 python-mpi/3.12 intel-oneapi-mkl llvm/19 eigen mpfr
         module load triqs/3_unst_nix2.4_llvm ;;
  *) echo "usage: sbatch $0 cthyb|ctseg|ctint" >&2; exit 2 ;;
esac

NRANKS=96
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/hubbard_holstein

# Two couplings, for two different jobs.
#
# G_MAIN = 0.7 (polaron shift g^2/omega_0^2 = 0.49, i.e. 0.12 U at U = 4) is the physically
# interesting point: a clearly visible retarded effect, still short of the bipolaronic
# regime. CTHYB and CTSEG handle it with sign 1.0.
#
# CTINT cannot. It expands in the full interaction including the retarded attractive
# density-density term, and its sign collapses as the polaron shift grows. Measured at
# beta = 10, U = 4, 15k cycles:
#
#     g      g^2/w0^2    sign     <k>
#     0.0    0.00        1.000    6.3     (pure Hubbard: no retarded term at all)
#     0.2    0.04        0.908    6.6
#     0.3    0.09        0.786    6.9
#     0.4    0.16        0.544    7.6
#     0.5    0.25        0.225    9.2
#     0.7    0.49        0.016   13.6     <- unusable
#
# So CTINT runs only at G_WEAK = 0.3, where all three solvers work and any disagreement is
# a genuine bug rather than a sign-problem artefact. That three-way check at weak coupling
# plus the CTHYB/CTSEG pair at G_MAIN covers both questions: "do the solvers agree?" and
# "what does a real retarded interaction do?".
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
MAX_TIME=1200
case "$SOLVER" in
  cthyb) NC_B10=500000;  NC_B100=50000  ;;
  ctseg) NC_B10=2000000; NC_B100=200000 ;;
  ctint) NC_B10=3000000; NC_B100=300000 ;;
esac

# beta = 100 needs a finer bosonic grid than beta = 10. K'(0) comes from Simpson quadrature
# on the n_tau_bosonic grid and the boson kernel is most curved at tau = 0 and beta, so the
# error grows with omega_0*beta/2: measured relative error at n_tau_bosonic = 2001 is 3e-12
# at omega_0*beta/2 = 5 but 3e-8 at 50 and 6e-7 at 100. Since K'(0) sets mu, give beta = 100
# the finer grid; it costs nothing.
GRID_B10="--n_tau 4096  --n_tau_bosonic 2001 --n_iw 1025"
GRID_B100="--n_tau 16384 --n_tau_bosonic 8001 --n_iw 2049"

# ---------------------------------------------------------------------------------------
# Chemical potentials
# ---------------------------------------------------------------------------------------
# Half filling is exact and needs no value: mu = U/2 - g^2/omega_0^2 = 1.51 here, derived
# from static_shift/half_filling_mu and asserted by every run.
#
# Calibrated 2026-09-18 with calibrate_mu.py (CTSEG probe, 20k cycles/point, U=4, g=0.7,
# omega_0=1, semicircular bath). Achieved n per spin-orbital is quoted.
#   python calibrate_mu.py --beta 10  --target_n 0.75 $MODEL
#   python calibrate_mu.py --beta 100 --target_n 0.75 $MODEL
#
# Unlike the spin-spin benchmark, n(mu) here is gentle and almost beta-independent
# (3.4269 at beta=10 vs 3.4130 at beta=100, a 0.4% difference), because the Holstein
# coupling shifts the level uniformly rather than opening a low-energy feature.
# The probe reproduces the exact half-filling mu to 1e-4, which is a free check on the
# closed form mu = U/2 - g^2/omega_0^2 = 1.51.
MU_B10_N075=3.426927    # -> n = 0.75110
MU_B100_N075=3.412964   # -> n = 0.75053

run () {  # run <beta> <n_cycles> <grid> [extra...]
  local beta="$1"; local ncyc="$2"; local grid="$3"; shift 3
  echo "=== $SOLVER  beta=$beta  n_cycles=$ncyc  $* ==="
  mpirun -n "$NRANKS" python "run_${SOLVER}.py" $MODEL $grid \
      --beta "$beta" --n_cycles "$ncyc" --n_warmup_cycles $((ncyc / 20)) \
      --max_time "$MAX_TIME" "$@"
}

# ---------------------------------------------------------------- core grid: half filling
if [ "$SOLVER" = ctint ]; then
  # Weak coupling only -- see the sign table above. Extra cycles to pay for sign ~0.79.
  MODEL="$MODEL_BASE --g $G_WEAK"
  run 10  $((NC_B10 * 2))  "$GRID_B10"  --filling 0.5
  run 100 $((NC_B100 * 2)) "$GRID_B100" --filling 0.5
else
  run 10  "$NC_B10"  "$GRID_B10"  --filling 0.5
  run 100 "$NC_B100" "$GRID_B100" --filling 0.5
  # The same weak-coupling point, so CTHYB and CTSEG can be compared against CTINT there.
  MODEL_MAIN="$MODEL"; MODEL="$MODEL_BASE --g $G_WEAK"
  run 10 "$NC_B10" "$GRID_B10" --filling 0.5
  MODEL="$MODEL_MAIN"
fi

# ------------------------------------------------------------------ core grid: n = 0.75
if [ -n "$MU_B10_N075" ]; then
  run 10 "$NC_B10" "$GRID_B10" --filling 0.75 --mu "$MU_B10_N075"
else
  echo "SKIPPING beta=10 n=0.75: set MU_B10_N075 (see calibrate_mu.py)" >&2
fi
if [ -n "$MU_B100_N075" ]; then
  run 100 "$NC_B100" "$GRID_B100" --filling 0.75 --mu "$MU_B100_N075"
else
  echo "SKIPPING beta=100 n=0.75: set MU_B100_N075 (see calibrate_mu.py)" >&2
fi

# -------------------------------------------------- CTHYB internal cross-check (important)
# With uniform g every vertex is eligible for the analytic Lang-Firsov path, so lang_firsov
# True and False solve the same physics by two entirely different routes. Disagreement here
# is a solver bug, not statistics -- this is the single most informative run in the set.
# The stochastic route has a worse sign, hence the extra cycles.
if [ "$SOLVER" = cthyb ]; then
  run 10 $((NC_B10 * 2)) "$GRID_B10" --filling 0.5 --lang_firsov False
fi

# ---------------------------------------------------------------- extended (uncomment)
#run 100 $((NC_B100 * 2)) "$GRID_B100" --filling 0.5 --lang_firsov False
# Coupling sweep at the reference point, to see the retarded effect grow:
#for G in 0.3 0.5 0.9; do
#  mpirun -n "$NRANKS" python "run_${SOLVER}.py" --U 4.0 --g $G --omega_0 1.0 \
#      --bath semicircular --out_dir $OUT $GRID_B10 --beta 10 --filling 0.5 \
#      --n_cycles "$NC_B10" --n_warmup_cycles $((NC_B10 / 20)) --max_time "$MAX_TIME"
#done
