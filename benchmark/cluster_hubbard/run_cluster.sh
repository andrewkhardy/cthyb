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
# Cluster DCA / CDMFT of the square-lattice Hubbard model, CTHYB (cluster_cthyb.py) against CTINT (cluster_ctint.py).
#   usage:  sbatch run_cluster.sh cthyb|ctint CLUSTER [extra driver arguments]     (from this directory)
# CLUSTER is a name of SUPERLATTICES in clusters.py (2x2, 8A, 4x4, ...). 8A is the 8-site Betts cluster, the largest
# for which CTHYB is practical.
# Every solve runs for MAX_TIME seconds (n_cycles is only a cap), so the two solvers get the same time per iteration:
# compare the noise of Sigma, the sign and the order across iterations.
# cthyb: DCA and CDMFT at the driver's pauli_prob = 0.5. On 2x2 it sampled the same sign, order and Sigma as 0, with
#        the same or (CDMFT) half the autocorrelation time. Its cycles are about one autocorrelation time long: on 2x2,
#        100-move cycles gave ~2 cycles for DCA and ~9 for CDMFT, hence 250 and 1000 moves, with the warmup kept at
#        5e5 moves. DCA runs in each basis of BASES: waves (Delta(K) diagonal, U scattering the momenta) and sites
#        (Delta_RR' off-diagonal, U local), both by default on 2x2 to compare their cost, waves only on larger clusters,
#        where the site basis has one block of N sites per spin, like CDMFT on 8A below. The site basis runs one
#        iteration, from the same Hartree G0 as the waves: a local 2x2 test gave it a sign of ~0.01 there (0.99 in the
#        waves), so its loop would not go anywhere. CDMFT runs in the waves.
#        For 4x4 it bounds the memory of the local problem first and stops with the numbers if the node cannot hold it;
#        pass --force to attempt it anyway. CDMFT beyond 2x2 takes blocks of several cluster-momentum waves. On 8A a move
#        took ~0.3 s on 8 threads locally, so its warmup is cut to 200 moves (5000 x 100 would never end) and it
#        measures every 10 moves.
# ctint: DCA and CDMFT, in the site basis.
# Later arguments win, so e.g. "--max_time 1800 --U 4" overrides the settings below.
# SCHEMES picks the schemes (default both), e.g. "SCHEMES=DCA sbatch run_cluster.sh cthyb 8A", to split a set over
# jobs, and BASES the DCA bases, e.g. "BASES=sites". CTHYB CDMFT on 8A is one block of all 8 waves per spin: 81
# invariant subspaces, and the atomic problem took 494 s to build locally (26 s for DCA), once per iteration.

set -euo pipefail
SOLVER="${1:-}"
MODE="${2:-}"
case "$SOLVER/$MODE" in
  cthyb/?*|ctint/?*) shift 2 ;;
  *) echo "usage: sbatch $0 cthyb|ctint CLUSTER [extra driver arguments]" >&2; exit 2 ;;
esac

module load modules/2.5-20261005
module load triqs/multiorbital

NRANKS=96
MAX_TIME=900          # seconds per solve
N_LOOPS=12            # 8 left the 2x2 loop still converging by ~2x per iteration
N_CYCLES=1000000000   # per rank; a cap, MAX_TIME ends each solve
MODEL="--U 6.0 --beta 10.0 --n_loops $N_LOOPS --n_cycles $N_CYCLES --n_warmup_cycles 5000 --max_time $MAX_TIME"
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/cluster_hubbard
mkdir -p "$OUT"

run () {  # run <driver> <scheme> [extra...]; a failed run does not stop the next one
  local driver="$1" scheme="$2"; shift 2
  echo "=== $driver $MODE $scheme  $* ==="
  mpirun -n "$NRANKS" python "$driver" --cluster "$MODE" --scheme "$scheme" $MODEL --out_dir "$OUT" "$@" \
    || echo "=== $driver $MODE $scheme exited with $? ==="
}

WARMUP=""
if [ "$SOLVER/$MODE" = cthyb/8A ]; then
  WARMUP="--n_warmup_cycles 20 --length_cycle 10"
fi
if [ "$MODE" = 2x2 ]; then
  BASES="${BASES:-waves sites}"
else
  BASES="${BASES:-waves}"
fi

for scheme in ${SCHEMES:-DCA CDMFT}; do
  if [ "$SOLVER" = cthyb ]; then
    case "$scheme" in
      DCA)   CYCLE="--length_cycle 250 --n_warmup_cycles 2000"; SCHEME_BASES="$BASES" ;;
      CDMFT) CYCLE="--length_cycle 1000 --n_warmup_cycles 500"; SCHEME_BASES="waves" ;;
      *)     CYCLE=""; SCHEME_BASES="waves" ;;
    esac
    for basis in $SCHEME_BASES; do
      LOOPS=""
      if [ "$basis" = sites ]; then LOOPS="--n_loops 1"; fi
      run cluster_cthyb.py "$scheme" --basis "$basis" $LOOPS $CYCLE $WARMUP "$@"
    done
  else
    run cluster_ctint.py "$scheme" "$@"
  fi
done
