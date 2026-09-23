#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=ccq
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#
# Ergodicity diagnostic for the beta = 100, n = 0.75 CTHYB/CTSEG density disagreement
# (0.707 vs 0.751 at the same mu). Not a production run: output goes to $OUT/chains.
#
# The pert_order_dyn histogram there is bimodal -- a k_dyn ~ 0 mode (no local moment) and a
# k_dyn ~ 12 mode shaped like the half-filling one (moment present over all of beta) -- and
# the two solvers differ only in the relative weight of the modes. Within the moment sector
# (half filling) they agree bin for bin. So the question is whether each chain mixes between
# the modes, or thermalises into one and stays: a production rank is exactly one such chain,
# and the 96-rank average then reports where the chains landed, not the equilibrium weight.
#
# Test: many independent serial chains, each as long as one production rank, each with its
# own seed (the solvers' default seed depends only on the rank, so serial runs without
# --seed would all be the same chain). A chain that mixes gives a density close to the
# ensemble mean; chains that do not scatter, typically bimodally. Per solver:
#
#   24 chains  beta = 100, n = 0.75    the cell in question
#   12 chains  beta = 100, n = 0.5     moment sector only, where the solvers already agree
#   12 chains  beta = 10,  n = 0.75    control: unimodal histogram, solvers agree
#
# 2 solvers x 48 chains = 96 = one node, all concurrent, so wall time is one MAX_TIME.
#
# Analyse with:  python plot_chains.py     (knobs hardcoded at the top of that file)

set -euo pipefail
module load modules/2.5-beta1
module load triqs/multiorbital

OUT=/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin/chains
mkdir -p "$OUT/logs"
MODEL="--U 4.0 --J 1.0 --bath dmft --out_dir $OUT --jperp 1 --szsz 1"
MAX_TIME=1200
MU_B10_N075=4.335938    # same pinned values as run_spin_spin.sh
MU_B100_N075=4.068123

# Per-rank cycle counts of the production runs, so one chain == one production rank.
declare -A NC=( [cthyb_10]=500000 [cthyb_100]=25000 [ctseg_10]=2000000 [ctseg_100]=100000 )

chain () {  # chain <solver> <beta> <n_chains> <first_seed> [extra args...]
  local solver="$1" beta="$2" count="$3" seed0="$4"; shift 4
  local ncyc="${NC[${solver}_${beta}]}"
  for ((i = 0; i < count; i++)); do
    local seed=$((seed0 + i))
    srun --exact -n 1 -c 1 python "run_${solver}.py" $MODEL \
        --beta "$beta" --n_cycles "$ncyc" --n_warmup_cycles $((ncyc / 20)) \
        --max_time "$MAX_TIME" --seed "$seed" "$@" \
        > "$OUT/logs/${solver}_b${beta}_seed${seed}.log" 2>&1 &
  done
}

for solver in cthyb ctseg; do
  chain "$solver" 100 24 1000 --filling 0.75 --mu "$MU_B100_N075"
  chain "$solver" 100 12 2000 --filling 0.5
  chain "$solver" 10  12 3000 --filling 0.75 --mu "$MU_B10_N075"
done
wait
echo "done: $(ls "$OUT"/*.h5 | wc -l) chain files in $OUT"
