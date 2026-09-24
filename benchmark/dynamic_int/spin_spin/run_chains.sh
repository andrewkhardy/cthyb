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
# Round 5 (chains5), 24 chains x 20 min each, all means agreeing with CTSEG's 0.7475(26):
#   CTHYB, length_cycle 500, uniform vertex proposal      <n> spread 0.0175, 110k moves/s
#   CTHYB, length_cycle 500, local vertex proposal (p=0.75) spread 0.0179, 106k moves/s --
#          vertex acceptance 0.7% -> 10%, no gain in mixing, so the local proposal was removed
#   CTHYB, length_cycle 2000                               spread 0.0118, 160k moves/s
#   CTSEG (24 x ~2 min)                                     spread 0.0127
# O_tau insertions linear in the order (was order^2): O_tau went from 47% of the run to 1-3%,
# and <SzSz>(beta/2) agrees three ways (O_tau, kink estimator, CTSEG).
#
# This round: the settings kept, with more chains -- CTSEG 24 | CTHYB length_cycle 2000 72.
# Spin flip on.
#
# Seeds are spaced by 2: TRIQS's default RNG (RandMT) forces the seed odd, so 2k and 2k+1
# are the same chain.
#
# Analyse with:  python plot_chains.py     (knobs hardcoded at the top of that file)

set -euo pipefail
module load modules/2.5-beta1
module load triqs/multiorbital

OUT=/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin/chains6
rm -rf "$OUT"
mkdir -p "$OUT/logs"
MODEL="--U 4.0 --J 1.0 --bath dmft --out_dir $OUT --beta 100 --jperp 1 --szsz 1 --filling 0.75 --mu 4.068123"
MAX_TIME=1200

# CTHYB: the cycle cap deliberately unreachable (MAX_TIME stops it); warmup 2.5M moves either way.
CTHYB_ARGS="--n_cycles 1000000 --move_double False --spin_flip_move True"
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

chain ctseg 24 5000
chain cthyb 72 5200 $LC2000
wait
echo "done: $(ls "$OUT"/*.h5 | wc -l) chain files in $OUT"
