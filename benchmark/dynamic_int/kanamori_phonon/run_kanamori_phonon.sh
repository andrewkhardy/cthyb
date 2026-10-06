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
# Two-orbital Kanamori + Holstein phonon (model.py): CTHYB against exact diagonalization.
# CTSEG and CTINT lack the spin-flip and pair-hopping terms, so ED is the only reference.
#
#   usage:  sbatch run_kanamori_phonon.sh cthyb|ed
#
# The ED references take seconds; where triqs is already loaded, call run_ed.py directly.
# Plot with:  python plot_kanamori_phonon.py

set -euo pipefail
SOLVER="${1:-}"
case "$SOLVER" in
  cthyb|ed) ;;
  *) echo "usage: sbatch $0 cthyb|ed" >&2; exit 2 ;;
esac

module load modules/2.5-beta1
module load triqs/multiorbital

NRANKS=96
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/kanamori_phonon
ED_OUT=$OUT

# Unequal g puts part of the coupling in the stochastic residual; uniform g couples only to
# N_up and N_down and goes entirely through Lang-Firsov.
MODEL="--U 2.0 --J 0.3 --V 0.7 --eps_bath 0.0 --omega_0 1.0 --g 0.7 0.3"
MODEL_UNIFORM="--U 2.0 --J 0.3 --V 0.7 --eps_bath 0.0 --omega_0 1.0 --g 0.5 0.5"

# Chemical potentials. Half filling is model.py's default. n = 0.75 from the exact ED probe:
#   python calibrate_mu.py --beta 10  --target_n 0.75 --g 0.7 0.3
MU_B10_N075=3.624982    # -> n = 0.750000 (exact)
MU_B100_N075=3.646802   # -> n = 0.750000 (exact)

# ---------------------------------------------------------------------------------- ED
if [ "$SOLVER" = ed ]; then
  for BETA in 10.0 100.0; do
    python run_ed.py $MODEL --beta $BETA --n_ph 24 --n_ph_check 6 --out_dir "$ED_OUT"
    python run_ed.py $MODEL_UNIFORM --beta $BETA --n_ph 24 --n_ph_check 6 --out_dir "$ED_OUT"
    [ -n "$MU_B10_N075" ] && [ "$BETA" = 10.0 ] && \
      python run_ed.py $MODEL --beta 10.0  --mu "$MU_B10_N075"  --n_ph 24 --out_dir "$ED_OUT"
    [ -n "$MU_B100_N075" ] && [ "$BETA" = 100.0 ] && \
      python run_ed.py $MODEL --beta 100.0 --mu "$MU_B100_N075" --n_ph 24 --out_dir "$ED_OUT"
  done
  echo "ED references written to $ED_OUT"
  exit 0
fi

# ------------------------------------------------------------------------------- CTHYB
# n_cycles is per rank. max_time bounds warmup and accumulation together, so the job's worst
# case is the sum over the run calls below: 3 x 1800 s + 2 x 5400 s = 4.5 h.
MAX_TIME=1800
NC_B10=500000

# beta = 100: the auto-correlation time exceeds 1e4 cycles of 100 moves (~20 at beta = 10).
# Check the "equilibrium:" line run_cthyb.py prints before reading Sigma.
LC_B100=5000         # measuring D0 every 100 moves took 87% of the wall-clock
NW_B100=4000         # 2e7 warmup moves, ~5x the largest auto-correlation time seen (4e6 moves)
NC_B100=13000        # about what fits after warmup at ~0.33 s per cycle
MAX_TIME_B100=5400

run () {  # run <beta> <n_cycles> [extra...]
  local beta="$1"; local ncyc="$2"; shift 2
  local nwarm=$((ncyc / 20)) lcyc=100 tmax=$MAX_TIME
  if [ "$beta" = 100 ]; then nwarm=$NW_B100 lcyc=$LC_B100 tmax=$MAX_TIME_B100; fi
  echo "=== cthyb  beta=$beta  n_cycles=$ncyc  n_warmup=$nwarm  length_cycle=$lcyc  max_time=$tmax  $* ==="
  mpirun -n "$NRANKS" python run_cthyb.py --beta "$beta" --n_cycles "$ncyc" \
      --n_warmup_cycles "$nwarm" --length_cycle "$lcyc" --max_time "$tmax" --out_dir "$OUT" "$@"
}

# ------------------------------------------------- core grid: orbital-dependent coupling
#run 10  "$NC_B10"  $MODEL
run 100 "$NC_B100" $MODEL

# n = 0.75
if [ -n "$MU_B10_N075" ]; then
  run 10 "$NC_B10" $MODEL --mu "$MU_B10_N075"
else
  echo "SKIPPING beta=10 n=0.75: set MU_B10_N075 (see calibrate_mu.py)" >&2
fi
if [ -n "$MU_B100_N075" ]; then
  run 100 "$NC_B100" $MODEL --mu "$MU_B100_N075"
else
  echo "SKIPPING beta=100 n=0.75: set MU_B100_N075 (see calibrate_mu.py)" >&2
fi

# --------------------------------------------------------- uniform g: the routing control
# All Lang-Firsov: tests the analytic path against ED
run 10 "$NC_B10" $MODEL_UNIFORM

# Lang-Firsov off: the same physics through the stochastic path
run 10 $((NC_B10 * 2)) $MODEL_UNIFORM --lang_firsov False

# ---------------------------------------------------------------- extended (uncomment)
#run 100 $((NC_B100 * 2)) $MODEL_UNIFORM --lang_firsov False
