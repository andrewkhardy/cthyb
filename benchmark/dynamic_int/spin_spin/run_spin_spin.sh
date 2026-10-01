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
# Single-orbital retarded spin-spin benchmark: CTHYB and CTSEG against CTINT (model: model.py).
#   usage:  sbatch run_spin_spin.sh cthyb|ctseg|ctint      (one job per solver)
# beta in {10, 100} x n in {0.5, 0.75} with the full S.S; the Jperp-only and Sz.Sz-only cases
# at beta = 10, half filling. Plot with plot_spin_spin.py.

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

# n_cycles is per rank. CTINT also expands in U, so it has a higher order and a sign < 1.
MAX_TIME=1200
case "$SOLVER" in
  cthyb) NC_B10=500000;  NC_B100=25000  ;;
  ctseg) NC_B10=2000000; NC_B100=100000 ;;
  ctint) NC_B10=3000000; NC_B100=150000 ;;
esac

# Half filling needs no mu: model.py uses the exact U/2. n = 0.75 from
#   python calibrate_mu.py --beta 10  --target_n 0.75
#   mpirun -n 16 python calibrate_mu.py --beta 100 --target_n 0.75
MU_B10_N075=4.335938    # -> n = 0.748761
MU_B100_N075=4.068123   # -> n = 0.749973

run () {  # run <beta> <n_cycles> [extra args...]
  local beta="$1"; local ncyc="$2"; shift 2
  echo "=== $SOLVER  beta=$beta  n_cycles=$ncyc  $* ==="
  mpirun -n "$NRANKS" python "run_${SOLVER}.py" $MODEL \
      --beta "$beta" --n_cycles "$ncyc" --n_warmup_cycles $((ncyc / 20)) \
      --max_time "$MAX_TIME" "$@"
}

run 10  "$NC_B10"  --filling 0.5 --jperp 1 --szsz 1
run 100 "$NC_B100" --filling 0.5 --jperp 1 --szsz 1

run 10 "$NC_B10" --filling 0.5 --jperp 1 --szsz 0     # spin-flip only
run 10 "$NC_B10" --filling 0.5 --jperp 0 --szsz 1     # Sz.Sz only

run 10  "$NC_B10"  --filling 0.75 --mu "$MU_B10_N075"  --jperp 1 --szsz 1
run 100 "$NC_B100" --filling 0.75 --mu "$MU_B100_N075" --jperp 1 --szsz 1
