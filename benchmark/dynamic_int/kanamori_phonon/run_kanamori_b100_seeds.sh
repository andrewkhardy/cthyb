#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=ccq
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#
# Independent CTHYB chains at beta = 100, half filling, g = (0.7, 0.3): is the orbital-0
# density miss against ED (0.53 vs exactly 0.5) statistics or a stuck chain?
#
#   usage:  sbatch run_kanamori_b100_seeds.sh
#
# Each chain gets its own seed, so its file carries a _seed-N suffix; plot_kanamori_phonon.py
# skips those, so the main results are untouched. First check each chain equilibrated: the
# log's "Auto-correlation time" well below its cycle count, and run_cthyb.py's
# "equilibrium:" line clean. Only then compare orbital 0's <n> across the chains:
#   scattered around 0.5          -> statistics, the fix is more cycles
#   on different sides of 0.5, or
#   all at the same wrong value   -> ergodicity, the fix is a global move
#
# The first run of this script (2026-09-28: length_cycle 100, warmup 10k cycles, 1 h) gave
# all three chains at the same wrong value, 0.52 -- but not from ergodicity: the
# auto-correlation time was >40k cycles, so every rank was still relaxing from the same
# empty configuration and the chains shared its bias. Hence the settings below; see the
# beta = 100 block in run_kanamori_phonon.sh for the numbers behind them.
#   length_cycle 5000   measuring D0 every 100 moves was 80% of the wall-clock
#   warmup 4000 cycles  2e7 moves, ~5x the auto-correlation lower bound (4e6 moves), ~20 min
#   n_cycles 14000      about what fits in the ~60 min left at ~0.25 s per cycle
#   n_l 100             |G_l| is still ~0.09 at l = 40 at this beta
# max_time bounds warmup and accumulation together, so MAX_TIME is the whole chain.
# Wall clock: 3 chains x 80 min = 4 h, plus ~1 min each of startup and post-processing,
# inside the 5 h above.

set -euo pipefail

module load modules/2.5-beta1
module load triqs/multiorbital

NRANKS=96
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/kanamori_phonon
MODEL="--U 2.0 --J 0.3 --V 0.7 --eps_bath 0.0 --omega_0 1.0 --g 0.7 0.3"
MAX_TIME=4800

for SEED in 1 2 3; do
  echo "=== cthyb  beta=100  seed=$SEED ==="
  mpirun -n "$NRANKS" python run_cthyb.py $MODEL \
      --beta 100 --n_cycles 14000 --n_warmup_cycles 4000 --length_cycle 5000 \
      --max_time "$MAX_TIME" --n_l 100 --random_seed "$SEED" --out_dir "$OUT"
done
