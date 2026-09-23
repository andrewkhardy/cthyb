#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=ccq
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#
# Two-orbital Hubbard-Kanamori + Holstein phonon: CTHYB against exact diagonalization.
# Model (shared by both sides, so they solve one Hamiltonian): model.py
#
#   usage:  sbatch run_kanamori_phonon.sh cthyb     # the QMC grid
#           sbatch run_kanamori_phonon.sh ed        # the ED references (seconds, but this
#                                                   # script module-loads, so in an env that
#                                                   # already has triqs just call run_ed.py
#
# CTSEG and CTINT do not appear here, and not for want of trying: neither supports the
# off-diagonal Kanamori terms (spin-flip and pair-hopping), so ED is the only reference.
# That is not a weakness -- the ED here is EXACT, because the model is *defined* with one
# bath site per spin-orbital rather than fitting a discretised bath to a continuous one.
# The only approximation is the phonon truncation, which run_ed.py convergence-tests
# against n_ph + n_ph_check and reports. beta = 100 makes truncation easier, not harder.
#
# Plot with:  python plot_kanamori_phonon.py

set -euo pipefail
SOLVER="${1:-}"
case "$SOLVER" in
  cthyb|ed) ;;
  *) echo "usage: sbatch $0 cthyb|ed" >&2; exit 2 ;;
esac

module load modules/2.5-beta1
module load triqs/multiorbital

NRANKS=96
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/kanamori_phonon
ED_OUT=$OUT

# g = (0.7, 0.3): orbital-dependent phonon coupling, the "U_tot + dU" case. This is the
# interesting one -- uniform g is exactly a coupling to N_up/N_down and goes entirely
# through the analytic Lang-Firsov path, whereas unequal g forces part of the coupling
# into the stochastic residual, which is the machinery actually under test.
MODEL="--U 2.0 --J 0.3 --V 0.7 --eps_bath 0.0 --omega_0 1.0 --g 0.7 0.3"
MODEL_UNIFORM="--U 2.0 --J 0.3 --V 0.7 --eps_bath 0.0 --omega_0 1.0 --g 0.5 0.5"

# ---------------------------------------------------------------------------------------
# Chemical potentials
# ---------------------------------------------------------------------------------------
# Half filling is exact: model.py derives the particle-hole symmetric mu including the
# phonon shift, mu_a = 0.5 sum_b W_ab - g_a^2/(2 omega_0^2) with
# W_ab = kanamori_ab - g_a g_b/omega_0^2. run_ed.py confirms <n_a> = 0.5 to 1e-15.
#
# Away from half filling the probe is the ED itself, so it is exact and cheap -- no
# statistics, no tolerance, no risk of a non-monotonic scan:
#   python calibrate_mu.py --beta 10  --target_n 0.75 --g 0.7 0.3
#   python calibrate_mu.py --beta 100 --target_n 0.75 --g 0.7 0.3
#
# Calibrated 2026-09-21, exact ED probe, both hitting n = 0.750000 with zero deviation.
# beta = 100 needs a slightly larger mu than beta = 10 (3.6468 vs 3.6250): the same mu
# gives a marginally lower density at low T, since the thermal smearing that was helping
# fill the level at beta = 10 is gone.
MU_B10_N075=3.624982    # -> n = 0.750000 (exact)
MU_B100_N075=3.646802   # -> n = 0.750000 (exact)

# ---------------------------------------------------------------------------------------
# ED references -- fast and exact (seconds), so run these first
# ---------------------------------------------------------------------------------------
if [ "$SOLVER" = ed ]; then
  for BETA in 10.0 100.0; do
    python run_ed.py $MODEL --beta $BETA --n_ph 24 --n_ph_check 6 --out_dir "$ED_OUT"
    python run_ed.py $MODEL_UNIFORM --beta $BETA --n_ph 24 --n_ph_check 6 --out_dir "$ED_OUT"
    [ -n "$MU_B10_N075" ] && [ "$BETA" = 10.0 ] && \
      python run_ed.py $MODEL --beta 10.0  --mu "$MU_B10_N075"  --n_ph 24 --out_dir "$ED_OUT"
    [ -n "$MU_B100_N075" ] && [ "$BETA" = 100.0 ] && \
      python run_ed.py $MODEL --beta 100.0 --mu "$MU_B100_N075" --n_ph 24 --out_dir "$ED_OUT"
  done
  echo "ED references written to $ED_OUT"
  exit 0
fi

# ---------------------------------------------------------------------------------------
# Statistics and wall-clock
# ---------------------------------------------------------------------------------------
# 45 min budget for the multiorbital benchmarks. n_cycles is per rank.
# Measured single-core at beta = 10, uniform g: sign 1.0, 16 analytic / 0 stochastic
# vertices. Unequal g pushes part of the coupling into the stochastic residual, so expect
# a worse sign there -- that is the point of the run, and why it gets more cycles.
#
# WALL-CLOCK BUDGET -- worst case is (number of `run` calls) x MAX_TIME plus startup and
# the final h5 write. Six calls are live (both betas at half filling, both at n = 0.75
# since MU_B*_N075 are set, and the two $MODEL_UNIFORM runs):
#
#   6 calls x 1800 s = 180 min  -> fits --time=04:00:00 with an hour spare.
#
# This script was the worst offender of the set: 6 x 2700 s = 270 min against a 180 min
# allocation, i.e. 90 min over, not merely tight. The symptom was that the two
# $MODEL_UNIFORM runs at the bottom -- the last calls made -- had no cthyb output on disk
# at all while the four above them did. If more calls are added, redo this arithmetic.
MAX_TIME=1800
NC_B10=500000
# Halved on 2026-09-22. This is the most expensive cell in the whole benchmark set:
# measured <k_dyn> is 31.5 at beta = 100 half filling and 82.7 at beta = 100, n = 0.75,
# against 3.4 and 8.3 at beta = 10 -- a 10x order increase, with cost per cycle growing
# with the order. The sign stays 1.000 throughout, so this is pure cost, not variance.
# If the n = 0.75 call still caps out at MAX_TIME, quarter this rather than halve it;
# each run reports its achieved count, so retune from that.
NC_B100=25000

run () {  # run <beta> <n_cycles> [extra...]
  local beta="$1"; local ncyc="$2"; shift 2
  echo "=== cthyb  beta=$beta  n_cycles=$ncyc  $* ==="
  mpirun -n "$NRANKS" python run_cthyb.py --beta "$beta" --n_cycles "$ncyc" \
      --n_warmup_cycles $((ncyc / 20)) --max_time "$MAX_TIME" --out_dir "$OUT" "$@"
}

# ------------------------------------------------- core grid: orbital-dependent coupling
run 10  "$NC_B10"  $MODEL
run 100 "$NC_B100" $MODEL

# n = 0.75, once calibrated
if [ -n "$MU_B10_N075" ]; then
  run 10 "$NC_B10" $MODEL --mu "$MU_B10_N075"
else
  echo "SKIPPING beta=10 n=0.75: set MU_B10_N075 (see calibrate_mu.py)" >&2
fi
if [ -n "$MU_B100_N075" ]; then
  run 100 "$NC_B100" $MODEL --mu "$MU_B100_N075"
else
  echo "SKIPPING beta=100 n=0.75: set MU_B100_N075 (see calibrate_mu.py)" >&2
fi

# --------------------------------------------------------- uniform g: the routing control
# Uniform g sends every vertex through Lang-Firsov (16 analytic, 0 stochastic), so this run
# isolates the analytic path against ED. Comparing it with the g = (0.7, 0.3) run above
# separates "is the analytic resummation right?" from "is the stochastic residual right?".
run 10 "$NC_B10" $MODEL_UNIFORM

# The same uniform model with the analytic path switched off: same physics, entirely
# different machinery. Disagreement here is a solver bug, not statistics.
run 10 $((NC_B10 * 2)) $MODEL_UNIFORM --lang_firsov False

# ---------------------------------------------------------------- extended (uncomment)
#run 100 $((NC_B100 * 2)) $MODEL_UNIFORM --lang_firsov False
