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
# Two-patch valence-bond dimer with a retarded site-local S.S interaction (model.py), whose
# vertices are off-diagonal in the solver's patch basis.
#
#   sbatch run_vb_dimer.sh cthyb     # the CTHYB runs
#   sbatch run_vb_dimer.sh ed        # the ED references
#   python check_rotation.py         # symbolic checks, no MC
#
# Two coupling points:
#   J_ED   J_intra = -J_inter: the nearest ED-representable point; ED is the exact reference.
#   J_DCA  J_intra = 0: the VBDMFT coupling. No ED exists (-J is indefinite), so the
#          cross-check is lang_firsov True vs False.
#
# Plot with plot_vb_dimer.py.

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
J_DCA="--J_intra 0.0 --J_inter 0.5 --bath dca"

# ---------------------------------------------------------------- chemical potentials
# mu = U/2 is half filling at $J_ED. The DCA bath is not particle-hole symmetric, so $J_DCA
# needs calibrating at n = 0.5 too; until then it runs at U/2.
#   python calibrate_mu.py --beta 10 --target_n 0.75 $J_ED
#   mpirun -n 16 python calibrate_mu.py --target_n 0.5 $J_DCA
# MU_B10_N075 came from an n_ph = 2 ED probe; the n_ph = 3 runs may sit slightly off
# n = 0.75, but ED and CTHYB share the mu, so they solve the same Hamiltonian.
MU_B10_N075=2.404877    # $J_ED,  n = 0.75
MU_B100_N075=""         # $J_ED,  n = 0.75, not yet calibrated
MU_DCA_B10_N05=""       # $J_DCA, n = 0.5
MU_DCA_B100_N05=""

# ------------------------------------------------------------------------ ED references
# Blocks are 70 x (n_ph+1)^3: ~1.9 h per point at --n_ph 3 with the n_ph check.
# (--n_ph 2 --n_ph_check 0 takes ~16 s, for a local test.)
if [ "$SOLVER" = ed ]; then
  for BETA in 10.0 100.0; do
    python run_ed.py $COMMON $J_ED --beta $BETA --n_ph 3 --n_ph_check 1 --out_dir "$OUT"
  done
  [ -n "$MU_B10_N075" ] && \
    python run_ed.py $COMMON $J_ED --beta 10.0  --mu "$MU_B10_N075"  --n_ph 3 --n_ph_check 1 --out_dir "$OUT"
  [ -n "$MU_B100_N075" ] && \
    python run_ed.py $COMMON $J_ED --beta 100.0 --mu "$MU_B100_N075" --n_ph 3 --n_ph_check 1 --out_dir "$OUT"
  echo "ED references written to $OUT"
  echo "NOTE: no ED for the J_DCA point -- see the header for why."
  exit 0
fi

# ---------------------------------------------------------------------------- CTHYB runs
# Worst-case wall clock is (number of run calls) x MAX_TIME: 7 x 30 min once every MU_* is set.
MAX_TIME=1800
NC_B10=500000
NC_B100=25000   # the expansion order is ~9x larger at beta = 100 than at beta = 10

run () {  # run <beta> <n_cycles> <model args...>
  local beta="$1"; local ncyc="$2"; shift 2
  echo "=== cthyb  beta=$beta  n_cycles=$ncyc  $* ==="
  mpirun -n "$NRANKS" python run_cthyb.py $COMMON --beta "$beta" --n_cycles "$ncyc" \
      --n_warmup_cycles $((ncyc / 20)) --max_time "$MAX_TIME" --out_dir "$OUT" "$@"
}

# J_ED, half filling
run 10  "$NC_B10"  $J_ED
run 100 "$NC_B100" $J_ED

# J_DCA, half filling (mu = U/2 until MU_DCA_* is set)
[ -z "$MU_DCA_B10_N05" ] && echo "NOTE: beta=10 DCA at mu = U/2, not calibrated to n = 0.5" >&2
run 10  "$NC_B10"  $J_DCA ${MU_DCA_B10_N05:+--mu "$MU_DCA_B10_N05"}
[ -z "$MU_DCA_B100_N05" ] && echo "NOTE: beta=100 DCA at mu = U/2, not calibrated to n = 0.5" >&2
run 100 "$NC_B100" $J_DCA ${MU_DCA_B100_N05:+--mu "$MU_DCA_B100_N05"}

# J_ED, n = 0.75
if [ -n "$MU_B10_N075" ]; then
  run 10 "$NC_B10" $J_ED --mu "$MU_B10_N075"
else
  echo "SKIPPING beta=10 n=0.75: set MU_B10_N075 (see calibrate_mu.py)" >&2
fi
if [ -n "$MU_B100_N075" ]; then
  run 100 "$NC_B100" $J_ED --mu "$MU_B100_N075"
else
  echo "SKIPPING beta=100 n=0.75: set MU_B100_N075 (see calibrate_mu.py)" >&2
fi

# J_DCA lang_firsov=False partner, at the same mu as the lf=True run above
run 10 $((NC_B10 * 2)) $J_DCA ${MU_DCA_B10_N05:+--mu "$MU_DCA_B10_N05"} --lang_firsov False

# ---------------------------------------------------------------- extended (uncomment)
#run 100 $((NC_B100 * 2)) $J_DCA --lang_firsov False
#
# Not worth running: at $J_ED every vertex is patch-off-diagonal (check_rotation.py), so
# lf=True and lf=False sample the same vertices and differ only by seed.
#run 10  $((NC_B10 * 2))  $J_ED  --lang_firsov False
