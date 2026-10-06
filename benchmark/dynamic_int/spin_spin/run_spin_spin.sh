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
# Single-orbital retarded spin-spin benchmark (model.py), one job per solver:
#
#   sbatch run_spin_spin.sh cthyb|ctseg|ctint
#
# Grid: beta in {10, 100} x n in {0.5, 0.75} with full S.S, plus the spin-flip-only and
# Sz.Sz-only cases at beta = 10, half filling. Plot with plot_spin_spin.py.

set -euo pipefail
SOLVER="${1:-}"
case "$SOLVER" in
  cthyb|ctseg|ctint) ;;
  *) echo "usage: sbatch $0 cthyb|ctseg|ctint" >&2; exit 2 ;;
esac

module load modules/2.5-beta1
module load triqs/multiorbital

NRANKS=96
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin
MODEL="--U 4.0 --J 1.0 --bath dmft --out_dir $OUT"

# Cycles per rank. CTINT expands in U too (sign ~0.37 at beta = 10), so it needs more; beta = 100
# costs far more per cycle. MAX_TIME caps each solve: worst case 6 runs x 20 min = 2 h.
MAX_TIME=1200
case "$SOLVER" in
  cthyb) NC_B10=500000;  NC_B100=25000  ;;
  ctseg) NC_B10=2000000; NC_B100=100000 ;;
  ctint) NC_B10=3000000; NC_B100=150000 ;;
esac

# Half filling is mu = U/2 (model.py). n = 0.75 is pinned from calibrate_mu.py (full S.S) so
# every solver runs the same model; n(mu) is steep at beta = 100, so it matters there.
MU_B10_N075=4.335938    # -> n = 0.748761  (deviation -1.2e-3)
MU_B100_N075=4.068123   # -> n = 0.749973  (deviation -2.7e-5)

run () {  # run <beta> <n_cycles> [extra args...]
  local beta="$1"; local ncyc="$2"; shift 2
  echo "=== $SOLVER  beta=$beta  n_cycles=$ncyc  $* ==="
  mpirun -n "$NRANKS" python "run_${SOLVER}.py" $MODEL \
      --beta "$beta" --n_cycles "$ncyc" --n_warmup_cycles $((ncyc / 20)) \
      --max_time "$MAX_TIME" "$@"
}

# Half filling, full S.S
run 10  "$NC_B10"  --filling 0.5 --jperp 1 --szsz 1
run 100 "$NC_B100" --filling 0.5 --jperp 1 --szsz 1

# The two pieces separately
run 10 "$NC_B10" --filling 0.5 --jperp 1 --szsz 0     # spin-flip only
run 10 "$NC_B10" --filling 0.5 --jperp 0 --szsz 1     # Sz.Sz only

# n = 0.75, full S.S
if [ -n "$MU_B10_N075" ]; then
  run 10  "$NC_B10"  --filling 0.75 --mu "$MU_B10_N075"  --jperp 1 --szsz 1
else
  echo "SKIPPING beta=10 n=0.75: set MU_B10_N075 (see calibrate_mu.py)" >&2
fi
if [ -n "$MU_B100_N075" ]; then
  run 100 "$NC_B100" --filling 0.75 --mu "$MU_B100_N075" --jperp 1 --szsz 1
else
  echo "SKIPPING beta=100 n=0.75: set MU_B100_N075 (see calibrate_mu.py)" >&2
fi

# Extended grid (uncomment as needed): the two pieces at beta = 100
#run 100 "$NC_B100" --filling 0.5 --jperp 1 --szsz 0
#run 100 "$NC_B100" --filling 0.5 --jperp 0 --szsz 1
#
# CTHYB without Lang-Firsov, as a cross-check; its sign is ~0.3 at J = 1, hence more cycles
#if [ "$SOLVER" = cthyb ]; then
#  run 10 $((NC_B10 * 4)) --filling 0.5 --jperp 1 --szsz 1 --lang_firsov False
#fi
