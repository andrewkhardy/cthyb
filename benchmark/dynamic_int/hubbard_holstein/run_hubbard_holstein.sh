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
# Single-orbital Hubbard-Holstein benchmark: CTHYB and CTSEG against CTINT (model: model.py).
#   usage:  sbatch run_hubbard_holstein.sh cthyb|ctseg|ctint      (one job per solver)
# beta in {10, 100} x n in {0.5, 0.75}, a weak-coupling point, and for CTHYB the
# lang_firsov=False cross-check. Plot with plot_hubbard_holstein.py.

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

G_MAIN=0.7   # polaron shift g^2/omega_0^2 = 0.49
G_WEAK=0.3   # the retarded term is a small correction to U
MODEL_BASE="--U 4.0 --omega_0 1.0 --bath semicircular --out_dir $OUT"
MODEL="$MODEL_BASE --g $G_MAIN"

# n_cycles is per rank.
MAX_TIME=1200
case "$SOLVER" in
  cthyb) NC_B10=500000;  NC_B100=50000  ;;
  ctseg) NC_B10=2000000; NC_B100=200000 ;;
  ctint) NC_B10=3000000; NC_B100=300000 ;;
esac

# Half filling needs no mu: model.py uses U/2 - g^2/omega_0^2 (= 1.51 here). n = 0.75 from
#   python calibrate_mu.py --beta 10|100 --target_n 0.75
# The two are the same bisection grid point, not a copy-paste slip.
MU_B10_N075=3.416250    # -> n = 0.749263
MU_B100_N075=3.416250   # -> n = 0.751271

run () {  # run <beta> <n_cycles> [extra...]
  local beta="$1"; local ncyc="$2"; shift 2
  echo "=== $SOLVER  beta=$beta  n_cycles=$ncyc  $* ==="
  mpirun -n "$NRANKS" python "run_${SOLVER}.py" $MODEL \
      --beta "$beta" --n_cycles "$ncyc" --n_warmup_cycles $((ncyc / 20)) \
      --max_time "$MAX_TIME" "$@"
}

run 10  "$NC_B10"  --filling 0.5
run 100 "$NC_B100" --filling 0.5
MODEL="$MODEL_BASE --g $G_WEAK"
run 10  "$NC_B10"  --filling 0.5
MODEL="$MODEL_BASE --g $G_MAIN"

run 10  "$NC_B10"  --filling 0.75 --mu "$MU_B10_N075"
run 100 "$NC_B100" --filling 0.75 --mu "$MU_B100_N075"

# Uniform g routes every vertex analytically; forcing the stochastic path must agree.
if [ "$SOLVER" = cthyb ]; then
  run 10 $((NC_B10 * 2)) --filling 0.5 --lang_firsov False
fi
