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
<<<<<<< HEAD
#   usage:  sbatch run_cluster_4x4.sh cthyb|ctint 2x2|4x4|8A [extra driver arguments]     (from this directory)
=======
#   usage:  sbatch run_cluster_4x4.sh cthyb|ctint 2x2|8A|4x4 [extra driver arguments]     (from this directory)
# 8A is the 8-site Betts cluster (clusters.py), the largest for which CTHYB is practical.
>>>>>>> b88742fc005ae8c58dc190939b94d6d83039d9be
# Every solve runs for MAX_TIME seconds (n_cycles is only a cap), so the two solvers get the same time per iteration:
# compare the noise of Sigma, the sign and the order across iterations.
# cthyb: DCA and CDMFT at pauli_prob = 0 and 0.5. For 4x4 it bounds the memory of the local problem first and stops
#        with the numbers if the node cannot hold it; pass --force to attempt it anyway. CDMFT beyond 2x2 takes blocks
#        of several cluster-momentum waves. On 8A a move took ~0.3 s on 8 threads locally, so its warmup is cut to 200 moves
#        (5000 x 100 would never end) and it measures every 10 moves.
# ctint: DCA and CDMFT, in the site basis.
# The Betts cluster 8A runs DCA only.
# Later arguments win, so e.g. "--max_time 1800 --U 4" overrides the settings below.
# SCHEMES picks the schemes (default both), e.g. "SCHEMES=DCA sbatch run_cluster_4x4.sh cthyb 8A", to split a set over
# jobs. CTHYB CDMFT on 8A is one block of all 8 waves per spin: 81 invariant subspaces, and the atomic problem took 494 s
# to build locally (26 s for DCA), once per iteration.

set -euo pipefail
SOLVER="${1:-}"
MODE="${2:-}"
case "$SOLVER/$MODE" in
<<<<<<< HEAD
  cthyb/2x2|cthyb/4x4|cthyb/8A|ctint/2x2|ctint/4x4|ctint/8A) shift 2 ;;
  *) echo "usage: sbatch $0 cthyb|ctint 2x2|4x4|8A [extra driver arguments]" >&2; exit 2 ;;
esac
case "$MODE" in
  8A) SCHEMES="DCA" ;;
  *)  SCHEMES="DCA CDMFT" ;;
=======
  cthyb/2x2|cthyb/8A|cthyb/4x4|ctint/2x2|ctint/8A|ctint/4x4) shift 2 ;;
  *) echo "usage: sbatch $0 cthyb|ctint 2x2|8A|4x4 [extra driver arguments]" >&2; exit 2 ;;
>>>>>>> b88742fc005ae8c58dc190939b94d6d83039d9be
esac

module load modules/2.5-20261005
module load triqs/multiorbital

NRANKS=96
MAX_TIME=900          # seconds per solve
N_LOOPS=12            # 8 left the 2x2 loop still converging by ~2x per iteration
N_CYCLES=1000000000   # per rank; a cap, MAX_TIME ends each solve
MODEL="--U 6.0 --beta 10.0 --n_loops $N_LOOPS --n_cycles $N_CYCLES --n_warmup_cycles 5000 --max_time $MAX_TIME"
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/cluster_4x4
mkdir -p "$OUT"

run () {  # run <driver> <scheme> [extra...]; a failed run does not stop the next one
  local driver="$1" scheme="$2"; shift 2
  echo "=== $driver $MODE $scheme  $* ==="
  mpirun -n "$NRANKS" python "$driver" --cluster "$MODE" --scheme "$scheme" $MODEL --out_dir "$OUT" "$@" \
    || echo "=== $driver $MODE $scheme exited with $? ==="
}

<<<<<<< HEAD
for scheme in $SCHEMES; do
=======
WARMUP=""
if [ "$SOLVER/$MODE" = cthyb/8A ]; then
  WARMUP="--n_warmup_cycles 20 --length_cycle 10"
fi

for scheme in ${SCHEMES:-DCA CDMFT}; do
>>>>>>> b88742fc005ae8c58dc190939b94d6d83039d9be
  if [ "$SOLVER" = cthyb ]; then
    for pp in 0.0 0.5; do
      run cluster_4x4.py "$scheme" --pauli_prob "$pp" $WARMUP "$@"
    done
  else
    run cluster_ctint.py "$scheme" "$@"
  fi
done
