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
# (0.707 vs 0.751 at the same mu). Not a production run.
#
# Round 1 (output in chains/): one production rank's worth per chain. Every chain of both
# solvers fell on a single n-vs-mode-weight curve -- same Hamiltonian, different mixing --
# with CTHYB chains spread over n = 0.58..0.84 against CTSEG's 0.70..0.79. Timing showed
# why: CTHYB ran 78 cycles/s against CTSEG's ~4500, with 56% of CTHYB's time in the O_tau
# and D0_corr measurements (length_cycle = 100 is tiny next to hybridisation order ~44) and
# 40% of its move time in 4-operator moves accepting 0.05% of proposals. Its warmup was
# 1250 cycles = 6 s.
#
# Round 2 (output in chains2/): same question with CTHYB tuned -- length_cycle 500,
# move_double off, a warmup of several mode-switching times, and n_cycles large enough
# that MAX_TIME is what stops it. CTSEG gets 5x its production cycles. If the tuned CTHYB
# chains tighten toward CTSEG's spread and the two means agree, the fix is parameters; if
# they stay wide, the Jperp insertion move needs to change (0.23% acceptance vs CTSEG 5.6%).
#
# SEEDS ARE SPACED BY 2. TRIQS's default RNG (RandMT, random_name = "") forces the seed odd
# with `seed | 1`, so seeds 2k and 2k+1 are the *same* chain -- round 1 had half the
# independent chains it appeared to. Production is unaffected: its default rank seeds
# 34788 + 928374 r are all even and stay distinct.
#
# Also compared: the new CTHYB moves (needs the rebuilt solver) -- local insert/remove of
# Jperp vertices with both ends in one operator-free stretch, and a full up <-> down swap
# that carries the vertices along -- against the tuned old move set, at the same seeds' worth
# of chains, so the difference in chain spread is the effect of the moves alone.
#
#   beta = 100, n = 0.75:  CTHYB old moves 16 | CTHYB new moves 24 | CTSEG 32
#   beta = 100, n = 0.5:   CTHYB new moves  8 | CTSEG 16
#
# 96 chains, all concurrent, so wall time is one warmup plus one MAX_TIME.
#
# Analyse with:  python plot_chains.py     (knobs hardcoded at the top of that file)

set -euo pipefail
module load modules/2.5-beta1
module load triqs/multiorbital

OUT=/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin/chains2
mkdir -p "$OUT/logs"
MODEL="--U 4.0 --J 1.0 --bath dmft --out_dir $OUT --jperp 1 --szsz 1"
MAX_TIME=1200
MU_B100_N075=4.068123   # same pinned value as run_spin_spin.sh

# CTHYB: 500 moves/cycle. The cycle cap is deliberately unreachable (MAX_TIME stops it);
# warmup 5000 cycles = 2.5M moves, about one round-1 chain in full.
CTHYB_ARGS="--length_cycle 500 --move_double False --n_cycles 1000000 --n_warmup_cycles 5000"
CTHYB_OLD="--move_dyn_local False --spin_flip_move False"
CTHYB_NEW="--move_dyn_local True --spin_flip_move True"
# CTSEG: 5x production, ~2 min per chain.
CTSEG_ARGS="--length_cycle 100 --n_cycles 500000 --n_warmup_cycles 25000"

chain () {  # chain <solver> <n_chains> <first_seed> [extra args...]
  local solver="$1" count="$2" seed0="$3"; shift 3
  local solver_args; [ "$solver" = cthyb ] && solver_args="$CTHYB_ARGS" || solver_args="$CTSEG_ARGS"
  for ((i = 0; i < count; i++)); do
    local seed=$((seed0 + 2 * i))
    srun --exact -n 1 -c 1 python "run_${solver}.py" $MODEL --beta 100 $solver_args \
        --max_time "$MAX_TIME" --seed "$seed" "$@" \
        > "$OUT/logs/${solver}_seed${seed}.log" 2>&1 &
  done
}

# Seed ranges keep the variants apart on disk (the filename carries the seed, not the moves)
chain cthyb 16 1000 --filling 0.75 --mu "$MU_B100_N075" $CTHYB_OLD
chain cthyb 24 1100 --filling 0.75 --mu "$MU_B100_N075" $CTHYB_NEW
chain ctseg 32 1000 --filling 0.75 --mu "$MU_B100_N075"
chain cthyb  8 2100 --filling 0.5 $CTHYB_NEW
chain ctseg 16 2000 --filling 0.5
wait
echo "done: $(ls "$OUT"/*.h5 | wc -l) chain files in $OUT"
