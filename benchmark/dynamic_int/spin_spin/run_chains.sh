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
# Independent-chain diagnostics, full S.S at beta = 100, mu(n = 0.75). Not a production run.
# One chain per core, each with its own seed.
#
# Round 4 (chains4, fixed build) brought CTHYB onto CTSEG everywhere: full S.S
# <n> = 0.7425(44) vs 0.7457(52), Sz.Sz only agreeing at beta = 10, 30, 100. This round asks
# two questions of the same cell, at equal wall time per chain:
#
#   1. does insert_dyn's local proposal help?   CTHYB p_local = 0     (move_dyn_local False)
#                                               CTHYB p_local = 0.75  (move_dyn_local True)
#   2. does measuring less often help?          CTHYB p_local = 0.75, length_cycle 2000. The
#                                               O_tau measurement was 47% of the run: it makes
#                                               (hybridisation order)^2 ~ 1600 insertions per
#                                               call at beta = 100 (O_tau_ins.cpp; min_ins is only
#                                               a floor), ~4 ms, i.e. as much as ~1200 moves
#
# Spin flip on in all three. 24 chains each, plus 24 CTSEG chains as the reference.
# Compare the chain spread of <n> and <k_dyn> between variants: at equal wall time a smaller
# spread is the improvement, and every mean must agree with CTSEG.
#
# Seeds are spaced by 2: TRIQS's default RNG (RandMT) forces the seed odd, so 2k and 2k+1
# are the same chain.
#
# Analyse with:  python plot_chains.py     (knobs hardcoded at the top of that file)

set -euo pipefail
module load modules/2.5-beta1
module load triqs/multiorbital

OUT=/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin/chains5
rm -rf "$OUT"
mkdir -p "$OUT/logs"
MODEL="--U 4.0 --J 1.0 --bath dmft --out_dir $OUT --beta 100 --jperp 1 --szsz 1 --filling 0.75 --mu 4.068123"
MAX_TIME=1200

# CTHYB: the cycle cap deliberately unreachable (MAX_TIME stops it); warmup 2.5M moves either way.
CTHYB_ARGS="--n_cycles 1000000 --move_double False --spin_flip_move True"
LC500="--length_cycle 500 --n_warmup_cycles 5000"
LC2000="--length_cycle 2000 --n_warmup_cycles 1250"
CTSEG_ARGS="--length_cycle 100 --n_cycles 500000 --n_warmup_cycles 25000"

chain () {  # chain <solver> <n_chains> <first_seed> [extra args...]
  local solver="$1" count="$2" seed0="$3"; shift 3
  local solver_args; [ "$solver" = cthyb ] && solver_args="$CTHYB_ARGS" || solver_args="$CTSEG_ARGS"
  for ((i = 0; i < count; i++)); do
    local seed=$((seed0 + 2 * i))
    srun --exact -n 1 -c 1 python "run_${solver}.py" $MODEL $solver_args \
        --max_time "$MAX_TIME" --seed "$seed" "$@" \
        > "$OUT/logs/${solver}_seed${seed}.log" 2>&1 &
  done
}

# seed ranges keep the CTHYB variants apart on disk
chain ctseg 24 5000
chain cthyb 24 5000 $LC500 --move_dyn_local False
chain cthyb 24 5100 $LC500 --move_dyn_local True
chain cthyb 24 5200 $LC2000 --move_dyn_local True
wait
echo "done: $(ls "$OUT"/*.h5 | wc -l) chain files in $OUT"
