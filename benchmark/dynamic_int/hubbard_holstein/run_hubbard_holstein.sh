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
# Single-orbital Hubbard-Holstein benchmark (model.py): CTHYB and CTSEG against CTINT.
# Runs beta in {10, 100} x n in {0.5, 0.75}, a weak-coupling point, and for CTHYB the
# lang_firsov=False cross-check. One job per solver:
#
#   sbatch run_hubbard_holstein.sh cthyb|ctseg|ctint
#
# Plot with:  python plot_hubbard_holstein.py

set -euo pipefail
SOLVER="${1:-}"
case "$SOLVER" in
  cthyb|ctseg|ctint) ;;
  *) echo "usage: sbatch $0 cthyb|ctseg|ctint" >&2; exit 2 ;;
esac

module load modules/2.5-beta1
module load triqs/multiorbital

NRANKS=96
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/hubbard_holstein

# G_MAIN: polaron shift g^2/omega_0^2 = 0.49 = 0.12 U, clearly retarded but not bipolaronic.
# G_WEAK: the retarded term is a small correction to U.
# CTINT runs G_MAIN with sign > 0.97 thanks to the signed alpha of common/ctint.py.
G_MAIN=0.7
G_WEAK=0.3
MODEL_BASE="--U 4.0 --omega_0 1.0 --bath semicircular --out_dir $OUT"
MODEL="$MODEL_BASE --g $G_MAIN"

# n_cycles is per MPI rank. Worst-case wall time is (number of run calls) x MAX_TIME,
# 6 x 1200 s = 2 h for cthyb.
MAX_TIME=1200
case "$SOLVER" in
  cthyb) NC_B10=500000;  NC_B100=50000  ;;
  ctseg) NC_B10=2000000; NC_B100=200000 ;;
  ctint) NC_B10=3000000; NC_B100=300000 ;;
esac

# Chemical potentials. Half filling needs none: model.py gives mu = U/2 - g^2/omega_0^2 = 1.51.
# n = 0.75 from calibrate_mu.py (CTSEG probe, 20k cycles per point):
#   python calibrate_mu.py --beta 10  --target_n 0.75
#   python calibrate_mu.py --beta 100 --target_n 0.75
# The two are equal on purpose: n(mu) barely depends on beta here, and the deterministic
# bisection lands both on the same grid point.
MU_B10_N075=3.416250    # -> n = 0.749263  (deviation -7.4e-4)
MU_B100_N075=3.416250   # -> n = 0.751271  (deviation +1.3e-3)

run () {  # run <beta> <n_cycles> [extra...]
  local beta="$1"; local ncyc="$2"; shift 2
  echo "=== $SOLVER  beta=$beta  n_cycles=$ncyc  $* ==="
  mpirun -n "$NRANKS" python "run_${SOLVER}.py" $MODEL \
      --beta "$beta" --n_cycles "$ncyc" --n_warmup_cycles $((ncyc / 20)) \
      --max_time "$MAX_TIME" "$@"
}

# ---------------------------------------------------------------- half filling
run 10  "$NC_B10"  --filling 0.5
run 100 "$NC_B100" --filling 0.5
# weak coupling
MODEL_MAIN="$MODEL"; MODEL="$MODEL_BASE --g $G_WEAK"
run 10 "$NC_B10" --filling 0.5
MODEL="$MODEL_MAIN"

# ---------------------------------------------------------------- n = 0.75
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

# ---------------------------------------------------------------- CTHYB cross-check
# lang_firsov=False samples the phonon stochastically; it must agree with the analytic route.
if [ "$SOLVER" = cthyb ]; then
  run 10 $((NC_B10 * 2)) --filling 0.5 --lang_firsov False
fi

# ---------------------------------------------------------------- extended (uncomment)
#run 100 $((NC_B100 * 2)) --filling 0.5 --lang_firsov False
# Coupling sweep at beta = 10:
#for G in 0.3 0.5 0.9; do
#  mpirun -n "$NRANKS" python "run_${SOLVER}.py" --U 4.0 --g $G --omega_0 1.0 \
#      --bath semicircular --out_dir $OUT --beta 10 --filling 0.5 \
#      --n_cycles "$NC_B10" --n_warmup_cycles $((NC_B10 / 20)) --max_time "$MAX_TIME"
#done
