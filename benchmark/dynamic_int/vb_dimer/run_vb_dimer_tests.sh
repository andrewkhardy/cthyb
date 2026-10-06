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
# Diagnostics for the CTHYB-vs-ED disagreement at the J_ED point, beta = 10. Both use
# n_l = 100, so Sigma from G_l is not Legendre-truncated.
#
#   sbatch run_vb_dimer_tests.sh cthyb
#   sbatch run_vb_dimer_tests.sh ed
#
#   test_nl100_rot-site   the benchmark itself; its ED is the main rot-site file, linked in.
#   test_nl100_rot-none   --rotation none: no patch-off-diagonal vertices. Its own ED.
#
# If CTHYB matches ED at rot-none but not at rot-site, the off-diagonal vertex path is at fault.
# Plot with plot_vb_dimer.py, SUBDIR and ROTATION set to match.

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

# Link the main rot-site ED file, if it exists yet
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
