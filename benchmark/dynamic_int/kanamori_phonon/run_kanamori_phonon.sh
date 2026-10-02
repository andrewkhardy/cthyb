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
# Two-orbital Hubbard-Kanamori + Holstein phonon: CTHYB against exact diagonalization (model: model.py).
#   usage:  sbatch run_kanamori_phonon.sh cthyb|ed|nntau
# The ED is exact up to the phonon truncation (one bath site per spin-orbital by construction).
# CTSEG and CTINT cannot take the spin-flip and pair-hopping terms. Plot with plot_kanamori_phonon.py.

set -euo pipefail
SOLVER="${1:-}"
case "$SOLVER" in
  cthyb|ed|nntau) ;;
  *) echo "usage: sbatch $0 cthyb|ed|nntau" >&2; exit 2 ;;
esac

module load modules/2.5-beta1
module load triqs/multiorbital

NRANKS=96
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/kanamori_phonon
ED_OUT=$OUT

# Unequal g leaves a residual for the stochastic expansion; uniform g goes entirely through
# Lang-Firsov, the routing control.
MODEL="--U 2.0 --J 0.3 --V 0.7 --eps_bath 0.0 --omega_0 1.0 --g 0.7 0.3"
MODEL_UNIFORM="--U 2.0 --J 0.3 --V 0.7 --eps_bath 0.0 --omega_0 1.0 --g 0.5 0.5"

# Half filling needs no mu: model.py includes the phonon shift. n = 0.75 from the exact ED probe:
#   python calibrate_mu.py --beta 10|100 --target_n 0.75 --g 0.7 0.3
MU_B10_N075=3.624982    # -> n = 0.750000
MU_B100_N075=3.646802   # -> n = 0.750000

if [ "$SOLVER" = ed ]; then
  for BETA in 10.0 100.0; do
    python run_ed.py $MODEL --beta $BETA --n_ph 24 --n_ph_check 6 --out_dir "$ED_OUT"
    python run_ed.py $MODEL_UNIFORM --beta $BETA --n_ph 24 --n_ph_check 6 --out_dir "$ED_OUT"
    if [ "$BETA" = 10.0 ]; then MU=$MU_B10_N075; else MU=$MU_B100_N075; fi
    python run_ed.py $MODEL --beta $BETA --mu "$MU" --n_ph 24 --out_dir "$ED_OUT"
  done
  echo "ED references written to $ED_OUT"
  exit 0
fi

# n_cycles is per rank; max_time bounds warmup and accumulation together.
MAX_TIME=1800
NC_B10=500000

# beta = 100 mixes slowly: long cycles put the time into moves rather than D0 measurements, and
# the warmup spans several auto-correlation times. Check the "equilibrium:" line run_cthyb.py
# prints before reading Sigma.
LC_B100=5000
NW_B100=4000
NC_B100=13000
MAX_TIME_B100=5400

run () {  # run <beta> <n_cycles> [extra...]
  local beta="$1"; local ncyc="$2"; shift 2
  local nwarm=$((ncyc / 20)) lcyc=100 tmax=$MAX_TIME
  if [ "$beta" = 100 ]; then nwarm=$NW_B100 lcyc=$LC_B100 tmax=$MAX_TIME_B100; fi
  echo "=== cthyb  beta=$beta  n_cycles=$ncyc  n_warmup=$nwarm  length_cycle=$lcyc  max_time=$tmax  $* ==="
  mpirun -n "$NRANKS" python run_cthyb.py --beta "$beta" --n_cycles "$ncyc" \
      --n_warmup_cycles "$nwarm" --length_cycle "$lcyc" --max_time "$tmax" --out_dir "$OUT" "$@"
}

# <n_a(tau) n_b(0)> for every pair of spin-orbitals, most of them not conserved, for uniform g (all
# Lang-Firsov). Against ED (the ed mode writes it); plot with plot_nn_tau.py.
if [ "$SOLVER" = nntau ]; then
  run 10 100000 $MODEL_UNIFORM --measure_nn_tau True --max_time -1
  exit 0
fi

run 100 "$NC_B100" $MODEL
run 10  "$NC_B10"  $MODEL --mu "$MU_B10_N075"
run 100 "$NC_B100" $MODEL --mu "$MU_B100_N075"

run 10 "$NC_B10" $MODEL_UNIFORM
# The analytic path switched off: same physics, different machinery.
run 10 $((NC_B10 * 2)) $MODEL_UNIFORM --lang_firsov False
