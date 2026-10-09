#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=ccq
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#
# Two-patch valence-bond dimer with a retarded real-space spin-spin interaction (model: model.py).
#   usage:  sbatch run_vb_dimer.sh cthyb|ed
# An ED exists only for -J positive semidefinite (run_ed.py), so CTHYB is checked against ED at
# J_ED and against its own lang_firsov True/False pair at the DCA point. Plot with plot_vb_dimer.py.

set -euo pipefail
SOLVER="${1:-}"
case "$SOLVER" in
  cthyb|ed) ;;
  *) echo "usage: sbatch $0 cthyb|ed" >&2; exit 2 ;;
esac

module load modules/2.5-20261005
module load triqs/multiorbital

NRANKS=96
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/vb_dimer

COMMON="--t 0.25 --tp 0.0 --U 2.0 --omega_0 1.0"
# ED-representable (-J rank 1), on the discrete bath, which has a finite Hamiltonian.
J_ED="--J_intra -0.5 --J_inter 0.5 --bath discrete --V 0.5 --eps_bath 0.0"
# The physical DCA point, on the coarse-grained bath; no ED.
J_DCA="--J_intra 0.0 --J_inter 0.5 --bath dca"

# Half filling at J_ED is exactly mu = U/2 (model.py's default). Otherwise from calibrate_mu.py:
#   python calibrate_mu.py --beta 10 --target_n 0.75 $J_ED                    (exact ED probe)
#   mpirun -n 16 python calibrate_mu.py --target_n 0.5|0.75 $J_DCA            (CTHYB probe)
# The dca bath breaks particle-hole symmetry, so J_DCA needs its mu even at half filling.
MU_B10_N075=2.404877    # $J_ED,  -> n = 0.750000 (ED probe at --n_ph 2)
MU_B100_N075=""         # $J_ED,  n = 0.75, not yet calibrated
MU_DCA_B10_N05=""       # $J_DCA, n = 0.50
MU_DCA_B100_N05=""

# Each --n_ph 3 solve is a dense eigh on ~4500-dimensional blocks: submit it.
if [ "$SOLVER" = ed ]; then
  for BETA in 10.0 100.0; do
    python run_ed.py $COMMON $J_ED --beta $BETA --n_ph 3 --n_ph_check 1 --out_dir "$OUT"
  done
  python run_ed.py $COMMON $J_ED --beta 10.0  --mu "$MU_B10_N075"  --n_ph 3 --n_ph_check 1 --out_dir "$OUT"
  [ -n "$MU_B100_N075" ] && \
    python run_ed.py $COMMON $J_ED --beta 100.0 --mu "$MU_B100_N075" --n_ph 3 --n_ph_check 1 --out_dir "$OUT"
  echo "ED references written to $OUT"
  echo "NOTE: no ED for the J_DCA point -- see the header for why."
  exit 0
fi

# n_cycles is per rank.
MAX_TIME=1800
NC_B10=500000
NC_B100=25000

run () {  # run <beta> <n_cycles> <model args...>
  local beta="$1"; local ncyc="$2"; shift 2
  echo "=== cthyb  beta=$beta  n_cycles=$ncyc  $* ==="
  mpirun -n "$NRANKS" python run_cthyb.py $COMMON --beta "$beta" --n_cycles "$ncyc" \
      --n_warmup_cycles $((ncyc / 20)) --max_time "$MAX_TIME" --out_dir "$OUT" "$@"
}

run 10  "$NC_B10"  $J_ED
run 100 "$NC_B100" $J_ED

# Until MU_DCA_* is calibrated these take mu = U/2, as does the lf=False run below.
[ -z "$MU_DCA_B10_N05" ] && echo "NOTE: beta=10 DCA at mu = U/2, not calibrated to n = 0.5" >&2
run 10  "$NC_B10"  $J_DCA ${MU_DCA_B10_N05:+--mu "$MU_DCA_B10_N05"}
[ -z "$MU_DCA_B100_N05" ] && echo "NOTE: beta=100 DCA at mu = U/2, not calibrated to n = 0.5" >&2
run 100 "$NC_B100" $J_DCA ${MU_DCA_B100_N05:+--mu "$MU_DCA_B100_N05"}

run 10 "$NC_B10" $J_ED --mu "$MU_B10_N075"
if [ -n "$MU_B100_N075" ]; then
  run 100 "$NC_B100" $J_ED --mu "$MU_B100_N075"
else
  echo "SKIPPING beta=100 n=0.75: set MU_B100_N075 (see calibrate_mu.py)" >&2
fi

# The lf True/False pair, the DCA point's only cross-check; both take the same mu. (At $J_ED it
# would be vacuous: every vertex there is patch-off-diagonal, so neither run uses Lang-Firsov.)
run 10 $((NC_B10 * 2)) $J_DCA ${MU_DCA_B10_N05:+--mu "$MU_DCA_B10_N05"} --lang_firsov False
