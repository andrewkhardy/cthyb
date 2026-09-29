#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=ccq
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#
# Isolating the CTHYB-vs-ED disagreement at the J_ED point (G off by ~8% at w ~ 2, beta = 10).
#
#   usage:  sbatch run_vb_dimer_tests.sh cthyb
#           sbatch run_vb_dimer_tests.sh ed
#
# Two tests, each in its own directory so nothing mixes with the main results or each other:
#
#   test_nl100_rot-site   the benchmark itself (--rotation site) with n_l = 100 instead of 50,
#                         so Sigma from G_l is not Legendre-truncated below w ~ 20 at beta = 10.
#                         ED reference: the main rot-site ED file, linked in (n_l is CTHYB-only,
#                         so it is the same reference).
#   test_nl100_rot-none   --rotation none: the interaction is local in the patch (working) basis,
#                         so there are no patch-off-diagonal c^dag_K c_K' vertices. Its own ED.
#
# Reading them: if CTHYB matches ED at rot-none but not at rot-site, the fault is in the
# off-diagonal (general 4-index) stochastic vertex path. If both disagree, it is elsewhere.
#
# Plot with plot_vb_dimer.py, SUBDIR = "test_nl100_rot-site" / "test_nl100_rot-none" and
# ROTATION = "site" / "none" to match.
#
# Wall clock: cthyb is 2 runs x MAX_TIME = 1 h; ed is one --n_ph 3 solve plus the --n_ph 4
# truncation check, ~2 h on the main run's timing.

set -euo pipefail
SOLVER="${1:-}"
case "$SOLVER" in
  cthyb|ed) ;;
  *) echo "usage: sbatch $0 cthyb|ed" >&2; exit 2 ;;
esac

module load modules/2.5-beta1
module load triqs/multiorbital

NRANKS=96
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/vb_dimer
COMMON="--t 0.25 --tp 0.0 --U 2.0 --omega_0 1.0"
J_ED="--J_intra -0.5 --J_inter 0.5 --bath discrete --V 0.5 --eps_bath 0.0"
BETA=10.0
N_L=100
MAX_TIME=1800
NC=500000

dir_for () { echo "$OUT/test_nl${N_L}_rot-$1"; }
mkdir -p "$(dir_for site)" "$(dir_for none)"

# The rot-site ED reference is the main run's file. Linked, not copied, so a rerun of the
# main ED is picked up; skipped with a note if it is not there yet.
for f in "$OUT"/ed_beta-${BETA}_*_Jintra--0.5_Jinter-0.5_*_mu-1.0_bath-V-0.5_rot-site_nph-3.h5; do
  if [ -e "$f" ]; then
    ln -sf "$f" "$(dir_for site)/"
  else
    echo "NOTE: no main rot-site ED file yet -- run 'sbatch run_vb_dimer.sh ed', it links in later" >&2
  fi
done

if [ "$SOLVER" = ed ]; then
  echo "=== ed  beta=$BETA  --rotation none ==="
  python run_ed.py $COMMON $J_ED --rotation none --beta "$BETA" --n_ph 3 --n_ph_check 1 \
      --out_dir "$(dir_for none)"
  exit 0
fi

for ROT in site none; do
  echo "=== cthyb  beta=$BETA  --rotation $ROT  --n_l $N_L ==="
  mpirun -n "$NRANKS" python run_cthyb.py $COMMON $J_ED --rotation "$ROT" --beta "$BETA" \
      --n_l "$N_L" --n_cycles "$NC" --n_warmup_cycles $((NC / 20)) --max_time "$MAX_TIME" \
      --out_dir "$(dir_for "$ROT")"
done
