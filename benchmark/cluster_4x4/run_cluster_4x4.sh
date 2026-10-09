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
# Cluster DCA / CDMFT of the square-lattice Hubbard model, CTHYB (cluster_4x4.py) against CTINT (cluster_ctint.py).
#   usage:  sbatch run_cluster_4x4.sh cthyb|ctint 2x2|4x4 [extra driver arguments]     (from this directory)
# Every solve runs for MAX_TIME seconds (n_cycles is only a cap), so the two solvers get the same time per iteration:
# compare the noise of Sigma, the sign and the order across iterations.
# cthyb: DCA and CDMFT at pauli_prob = 0 and 0.5. For 4x4 it bounds the memory of the local problem first and stops
#        with the numbers if the node cannot hold it; pass --force to attempt it anyway.
# ctint: DCA and CDMFT, in the site basis.
# Later arguments win, so e.g. "--max_time 1800 --U 4" overrides the settings below.

set -euo pipefail
SOLVER="${1:-}"
MODE="${2:-}"
case "$SOLVER/$MODE" in
  cthyb/2x2|cthyb/4x4|ctint/2x2|ctint/4x4) shift 2 ;;
  *) echo "usage: sbatch $0 cthyb|ctint 2x2|4x4 [extra driver arguments]" >&2; exit 2 ;;
esac
L="${MODE%x*}"

module load modules/2.5-beta1
module load triqs/multiorbital

NRANKS=96
MAX_TIME=900          # seconds per solve
N_LOOPS=8
N_CYCLES=1000000000   # per rank; a cap, MAX_TIME ends each solve
MODEL="--U 6.0 --beta 10.0 --n_loops $N_LOOPS --n_cycles $N_CYCLES --n_warmup_cycles 5000 --max_time $MAX_TIME"
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/cluster_4x4
mkdir -p "$OUT"

run () {  # run <driver> <scheme> [extra...]; a failed run does not stop the next one
  local driver="$1" scheme="$2"; shift 2
  echo "=== $driver ${L}x${L} $scheme  $* ==="
  mpirun -n "$NRANKS" python "$driver" --L "$L" --scheme "$scheme" $MODEL --out_dir "$OUT" "$@" \
    || echo "=== $driver ${L}x${L} $scheme exited with $? ==="
}

for scheme in DCA CDMFT; do
  if [ "$SOLVER" = cthyb ]; then
    for pp in 0.0 0.5; do
      run cluster_4x4.py "$scheme" --pauli_prob "$pp" "$@"
    done
  else
    run cluster_ctint.py "$scheme" "$@"
  fi
done
