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
# CTINT alpha scan: the benchmark points of the Hubbard-Holstein, spin-spin and cluster runs, solved with each alpha scheme
# of ../common/ctint.py for the same wall time, so the average order, the sign and the noise of G and Sigma compare
# directly. Summarize with summarize_ctint_alpha.py.
#   usage (from this directory):  sbatch run_ctint_alpha.sh hh|ss|cluster
# hh and ss: 4 and 5 points x 7 schemes, about 5 and 6 hours at MAX_TIME = 600. cluster: iteration 0 only (the Hartree
# start, the same G0 as the production runs) and no final solve, 2x2 DCA and CDMFT at five deltas and 4x4 at two, about
# 3 hours. At half filling 'centred' is 'signed' with a smaller delta (the density is 1/2), which is sign-free there for
# any delta. Every run tags its file with its alpha (--alpha_tag) and writes under $OUT. The summary compares the
# impurity points with the production runs and each cluster point with its own signed 0.51 run.
# SCHEMES="scheme:delta ..." replaces the scheme list, e.g. SCHEMES="centred:0.15"
# (SCHEMES_4X4 for the 4x4 cluster), and MAX_TIME the seconds per run.

set -euo pipefail
WHAT="${1:-}"
case "$WHAT" in
  hh|ss|cluster) ;;
  *) echo "usage: sbatch $0 hh|ss|cluster" >&2; exit 2 ;;
esac

module load modules/2.5-20261005
module load triqs/multiorbital

NRANKS=96
MAX_TIME="${MAX_TIME:-600}"   # seconds per run, the same for every scheme; n_cycles is only a cap
N_CYCLES=1000000000
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/ctint_alpha

case "$WHAT" in
  cluster) DEFAULT_SCHEMES="signed:0.51 centred:0.3 centred:0.2 centred:0.1 centred:0.05" ;;
  *)       DEFAULT_SCHEMES="signed:0.51 near:0.01 centred:0.3 centred:0.2 centred:0.1 centred:0.05 library:0.1" ;;
esac
SCHEMES="${SCHEMES:-$DEFAULT_SCHEMES}"

run () {  # run <driver> <out subdir> <scheme> <delta> [driver args...]; a failed run does not stop the next one
  local driver="$1" sub="$2" scheme="$3" delta="$4"; shift 4
  echo "=== $driver --alpha $scheme --alpha_delta $delta  $* ==="
  mkdir -p "$OUT/$sub"
  mpirun -n "$NRANKS" python "$driver" --alpha "$scheme" --alpha_delta "$delta" --alpha_tag \
      --n_cycles "$N_CYCLES" --max_time "$MAX_TIME" --out_dir "$OUT/$sub" "$@" \
    || echo "=== $driver --alpha $scheme --alpha_delta $delta exited with $? ==="
}

scan () {  # scan <driver> <out subdir> [driver args...]: every scheme at one point
  local driver="$1" sub="$2"; shift 2
  for sd in $SCHEMES; do
    run "$driver" "$sub" "${sd%%:*}" "${sd##*:}" "$@"
  done
}

case "$WHAT" in
  hh)  # as run_hubbard_holstein.sh; mu for n = 0.75 from its calibration
    HH="../hubbard_holstein/run_ctint.py"
    MODEL="--U 4.0 --omega_0 1.0 --bath semicircular --g 0.7 --n_warmup_cycles 2000"
    scan "$HH" hubbard_holstein $MODEL --beta 10  --filling 0.5
    scan "$HH" hubbard_holstein $MODEL --beta 10  --filling 0.75 --mu 3.416250
    scan "$HH" hubbard_holstein $MODEL --beta 100 --filling 0.5
    scan "$HH" hubbard_holstein $MODEL --beta 100 --filling 0.75 --mu 3.416250
    ;;
  ss)  # as run_spin_spin.sh; Sz.Sz only is where alpha sets every vertex (no Jperp)
    SS="../spin_spin/run_ctint.py"
    MODEL="--U 4.0 --J 1.0 --bath dmft --n_warmup_cycles 2000"
    scan "$SS" spin_spin $MODEL --beta 10  --filling 0.5  --jperp 1 --szsz 1
    scan "$SS" spin_spin $MODEL --beta 10  --filling 0.75 --mu 4.335938 --jperp 1 --szsz 1
    scan "$SS" spin_spin $MODEL --beta 100 --filling 0.5  --jperp 1 --szsz 1
    scan "$SS" spin_spin $MODEL --beta 100 --filling 0.75 --mu 4.068123 --jperp 1 --szsz 1
    scan "$SS" spin_spin $MODEL --beta 10  --filling 0.5  --jperp 0 --szsz 1
    ;;
  cluster)  # as run_cluster.sh, one iteration
    CL="../../cluster_hubbard/cluster_ctint.py"
    MODEL="--U 6.0 --beta 10.0 --n_loops 1 --final_time 0 --n_warmup_cycles 5000"
    for scheme in DCA CDMFT; do
      scan "$CL" cluster $MODEL --cluster 2x2 --scheme "$scheme"
    done
    SCHEMES="${SCHEMES_4X4:-signed:0.51 centred:0.1}"
    for scheme in DCA CDMFT; do
      scan "$CL" cluster $MODEL --cluster 4x4 --scheme "$scheme"
    done
    ;;
esac
