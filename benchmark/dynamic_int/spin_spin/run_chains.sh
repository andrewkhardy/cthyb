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
# Independent serial chains, full S.S at beta = 100, mu(n = 0.75): 24 CTSEG and 72 CTHYB
# chains, one per core, each with its own seed. A diagnostic, not a production run.
# Analyse with plot_chains.py.
#
# Seeds are spaced by 2: TRIQS's default RNG forces the seed odd, so 2k and 2k+1 are the same chain.

set -euo pipefail
module load modules/2.5-beta1
module load triqs/multiorbital

OUT=/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin/chains6
rm -rf "$OUT"
mkdir -p "$OUT/logs"
MODEL="--U 4.0 --J 1.0 --bath dmft --out_dir $OUT --beta 100 --jperp 1 --szsz 1 --filling 0.75 --mu 4.068123"
MAX_TIME=1200

# MAX_TIME ends the CTHYB chains, not n_cycles. Warmup is 2.5M moves for both solvers.
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
