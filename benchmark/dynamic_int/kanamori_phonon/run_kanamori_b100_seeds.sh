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
# Independent CTHYB chains at beta = 100, half filling, g = (0.7, 0.3): is a density miss
# against ED statistics or a stuck chain? Files carry a _seed-N suffix, which
# plot_kanamori_phonon.py skips.
#
#   usage:  sbatch run_kanamori_b100_seeds.sh
#
# Once each chain's "equilibrium:" line is clean, compare orbital 0's <n> across chains:
#   scattered around 0.5                     -> statistics: more cycles
#   all at one wrong value, or on both sides -> ergodicity: a global move
#
# Cycle settings as for beta = 100 in run_kanamori_phonon.sh; n_l 100 because |G_l| is still
# ~0.09 at l = 40. Wall clock: 3 chains x MAX_TIME = 4 h, inside --time.

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
