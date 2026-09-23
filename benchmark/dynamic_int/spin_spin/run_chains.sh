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
# Round 3 (chains3) located it: with Sz.Sz only -- Lang-Firsov, no stochastic vertices at
# all -- CTHYB gives <n> = 0.8509(6) against CTSEG's 0.8597(5) at the same mu, beta = 100.
# Jperp only (stochastic vertices, no Lang-Firsov) agrees with CTSEG. So the Lang-Firsov
# path, or something only it is sensitive to, is wrong at beta = 100. K(tau) grows like beta
# (max|K| = 4.3 against 0.43 at beta = 10), so anything harmless at beta = 10 may not be.
#
# This round, all Sz.Sz only, compared with CTSEG at the same mu:
#
#   beta = 100  CTSEG 8
#               CTHYB reference (dyn_n_l 50, length_cycle 500, no double moves) 12
#               CTHYB dyn_n_l 150                        -> Legendre truncation of K
#               CTHYB length_cycle 100 + double moves    -> the round-1 production settings
#               CTHYB lang_firsov False                  -> the same D0 sampled stochastically,
#                                                           no K(tau) at all (sign may be poor)
#   beta = 30   CTSEG 6 | CTHYB 6                        -> how the gap grows with beta
#   beta = 10   CTSEG 6 | CTHYB 6                        -> the baseline, away from half filling
#
# Seeds are spaced by 2: TRIQS's default RNG (RandMT) forces the seed odd, so 2k and 2k+1
# are the same chain.
#
# Analyse with:  python plot_chains.py     (knobs hardcoded at the top of that file)

set -euo pipefail
module load modules/2.5-beta1
module load triqs/multiorbital

OUT=/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin/chains4
rm -rf "$OUT"
mkdir -p "$OUT/logs"
MODEL="--U 4.0 --J 1.0 --bath dmft --out_dir $OUT --jperp 0 --szsz 1 --filling 0.75 --mu 4.068123"
MAX_TIME=1200

# CTHYB: the cycle cap deliberately unreachable (MAX_TIME stops it).
CTHYB_ARGS="--n_cycles 1000000 --move_dyn_local False --spin_flip_move False"
REF="--length_cycle 500 --move_double False --n_warmup_cycles 5000"
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

# beta = 100 (seed ranges keep the CTHYB variants apart on disk)
chain ctseg  8 1000 --beta 100
chain cthyb 12 1000 --beta 100 $REF
chain cthyb 12 1100 --beta 100 $REF --dyn_n_l 150
chain cthyb 12 1200 --beta 100 --length_cycle 100 --move_double True --n_warmup_cycles 25000
chain cthyb 28 1300 --beta 100 $REF --lang_firsov False
# beta = 30 and 10, same mu
chain ctseg  6 2000 --beta 30
chain cthyb  6 2000 --beta 30 $REF
chain ctseg  6 3000 --beta 10
chain cthyb  6 3000 --beta 10 $REF
wait
echo "done: $(ls "$OUT"/*.h5 | wc -l) chain files in $OUT"
