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
# Independent-chain diagnostics for the beta = 100 CTHYB/CTSEG disagreement. Not a
# production run. One chain per core, each with its own seed.
#
# Round 2 (chains2) showed, with the chains now mixing (spread 0.01):
#   * at n = 0.75 CTHYB sits at <n> = 0.70 against CTSEG's 0.75, 13 sigma apart, and the two
#     solvers lie on *different* n-vs-mode-weight lines -- a real difference, not mixing;
#     present with the old moves too
#   * at n = 0.5, CTHYB with the new moves has <k_dyn> = 9.6 against 11.8 for CTSEG and for
#     the old-code production run -- the new moves, or something else changed in round 2,
#     alter the sampled distribution
#
# This round isolates both, at beta = 100:
#
#   A  full S.S,   n = 0.5    CTHYB old moves 12 | local only 12 | spin flip only 12 | CTSEG 8
#      -> which new piece shifts <k_dyn> at half filling
#   B  Sz.Sz only, mu(n=0.75) CTHYB 16 | CTSEG 8
#      -> Lang-Firsov alone: CTHYB has no stochastic vertices at all here
#   C  Jperp only, mu(n=0.75) CTHYB old moves 12 | local only 8 | CTSEG 8
#      -> the stochastic vertex path alone (insert/remove/swap_dyn), no Lang-Firsov
#
# B and C are compared at the same mu, not the same density: whatever n comes out, the two
# solvers must agree on it. Whichever of B, C disagrees contains the n = 0.75 problem.
#
# Seeds are spaced by 2: TRIQS's default RNG (RandMT) forces the seed odd, so 2k and 2k+1
# are the same chain.
#
# Analyse with:  python plot_chains.py     (knobs hardcoded at the top of that file)

set -euo pipefail
module load modules/2.5-beta1
module load triqs/multiorbital

OUT=/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin/chains3
rm -rf "$OUT"
mkdir -p "$OUT/logs"
MODEL="--U 4.0 --J 1.0 --bath dmft --out_dir $OUT --beta 100"
MAX_TIME=1200
MU_B100_N075=4.068123   # same pinned value as run_spin_spin.sh

# CTHYB: 500 moves/cycle, the cycle cap deliberately unreachable (MAX_TIME stops it).
CTHYB_ARGS="--length_cycle 500 --move_double False --n_cycles 1000000 --n_warmup_cycles 5000"
OLD="--move_dyn_local False --spin_flip_move False"
LOCAL="--move_dyn_local True --spin_flip_move False"
FLIP="--move_dyn_local False --spin_flip_move True"
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

SS="--jperp 1 --szsz 1"
SZ="--jperp 0 --szsz 1 --filling 0.75 --mu $MU_B100_N075"
JP="--jperp 1 --szsz 0 --filling 0.75 --mu $MU_B100_N075"

# A: full S.S at half filling (seed ranges keep the move variants apart on disk)
chain cthyb 12 1000 $SS --filling 0.5 $OLD
chain cthyb 12 1100 $SS --filling 0.5 $LOCAL
chain cthyb 12 1200 $SS --filling 0.5 $FLIP
chain ctseg  8 1000 $SS --filling 0.5
# B: Sz.Sz only
chain cthyb 16 2000 $SZ $OLD
chain ctseg  8 2000 $SZ
# C: Jperp only
chain cthyb 12 3000 $JP $OLD
chain cthyb  8 3100 $JP $LOCAL
chain ctseg  8 3000 $JP
wait
echo "done: $(ls "$OUT"/*.h5 | wc -l) chain files in $OUT"
