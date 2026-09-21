#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=ccq
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#
# Single-orbital retarded spin-spin benchmark: CTHYB and CTSEG against CTINT.
# Model, conventions and the factors of 2: model.py
#
#   usage:  sbatch run_spin_spin.sh cthyb
#           sbatch run_spin_spin.sh ctseg
#           sbatch run_spin_spin.sh ctint
#
# One job per solver, so each gets its own wall-clock budget and a failure in one does not
# take the others down. (It used to be forced: the three needed incompatible module stacks.
# triqs/multiorbital now carries all three, so running them in one job would also work.)
#
# Grid: beta in {10, 100} x n in {0.5, 0.75}. The three coupling cases (full S.S, Jperp
# only, Sz.Sz only) are run at the reference point (beta = 10, half filling) only, since
# that is the validated one; the extended lines below run them everywhere.
#
# Plot with:  python plot_spin_spin.py     (knobs hardcoded at the top of that file)

set -euo pipefail
SOLVER="${1:-}"
case "$SOLVER" in
  cthyb|ctseg|ctint) ;;
  *) echo "usage: sbatch $0 cthyb|ctseg|ctint" >&2; exit 2 ;;
esac

# One stack for all three solvers: triqs/multiorbital now carries cthyb, ctseg and ctint.
module load modules/2.5-beta1
module load triqs/multiorbital

NRANKS=96
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin
MODEL="--U 4.0 --J 1.0 --bath dmft --out_dir $OUT"

# ---------------------------------------------------------------------------------------
# Statistics and wall-clock
# ---------------------------------------------------------------------------------------
# n_cycles is per MPI rank, so wall time is the single-rank time and the statistics improve
# with rank count. Halved from the old 1e6 at beta = 10, as asked.
#
# Measured single-core rates (beta = 10, U = 4, J = 1, full S.S, length_cycle = 50):
#   cthyb  2.4e4 cycles/s ... average order 2.36, sign 1.00
#   ctseg  3.4e4 cycles/s ... average order 2.42, sign 1.00   (~14x faster than cthyb)
#   ctint  1.3e3 cycles/s ... average order 13.5, sign 0.37
# CTINT expands in the full U as well, hence the much higher order and the sign problem; it
# needs roughly 1/sign^2 ~ 7x the statistics for comparable error bars, so it gets more
# cycles rather than the same number.
#
# beta = 100 is the cell at risk: cost per cycle grows with the perturbation order (itself
# roughly linear in beta), so expect 10-30x the beta = 10 cost per cycle. The values below
# are a first estimate -- MAX_TIME is the hard guarantee, and each run reports its achieved
# cycle count and error bars so these can be retuned. See the note in the plan about the
# budget being a target rather than a constraint.
MAX_TIME=1200          # 20 min hard cap inside the solver, per the agreed budget
case "$SOLVER" in
  cthyb) NC_B10=500000;  NC_B100=50000  ;;
  ctseg) NC_B10=2000000; NC_B100=200000 ;;
  ctint) NC_B10=3000000; NC_B100=300000 ;;
esac

# ---------------------------------------------------------------------------------------
# Chemical potentials
# ---------------------------------------------------------------------------------------
# Half filling needs no value: model.py derives the exact particle-hole symmetric mu = U/2
# from half_filling_mu (which includes the retarded coupling's K'(0) static shift), and the
# run asserts it.
#
# n = 0.75 per spin-orbital has no closed form, so it is calibrated once and pinned here so
# that every solver uses the identical mu -- without that, differences in Sigma would mix
# solver disagreement with a different model.
#
# Recalibrated 2026-09-21 with `calibrate_mu.py` (CTSEG probe, 20k cycles/point, U = 4,
# J = 1, full S.S, bath = dmft). The whole scan takes ~15 s.
#   python calibrate_mu.py --beta 10  --target_n 0.75
#   mpirun -n 16 python calibrate_mu.py --beta 100 --target_n 0.75
#
# The 2026-09-18 values (4.343276 / 4.078482) were produced before a bug was fixed in
# calibrate_mu.py: it reported an interpolated mu while quoting the *bisection's* achieved
# n, so the mu and the n in the comment came from different estimators and the quoted n
# never validated the pinned value. Both are now the bisection endpoint, with the n that
# endpoint actually achieved. beta = 10 moved by 0.007, beta = 100 by 0.010.
#
# Pinning these precisely matters more at beta = 100 than it looks: n(mu) is steep there
# (n went 0.726 -> 0.873 for a mu change of only 0.06, the low-temperature approach to a
# plateau edge), so a solver that found its own mu would land on a visibly different model.
# That steepness is also why the beta = 100 probe lands so much tighter than beta = 10 for
# the same tolerance -- a given density window corresponds to a narrower window in mu.
MU_B10_N075=4.335938    # -> n = 0.748761  (deviation -1.2e-3)
MU_B100_N075=4.068123   # -> n = 0.749973  (deviation -2.7e-5)

run () {  # run <beta> <n_cycles> [extra args...]
  local beta="$1"; local ncyc="$2"; shift 2
  echo "=== $SOLVER  beta=$beta  n_cycles=$ncyc  $* ==="
  mpirun -n "$NRANKS" python "run_${SOLVER}.py" $MODEL \
      --beta "$beta" --n_cycles "$ncyc" --n_warmup_cycles $((ncyc / 20)) \
      --max_time "$MAX_TIME" "$@"
}

# ---------------------------------------------------------------------------------------
# Core grid
# ---------------------------------------------------------------------------------------
# Half filling, both temperatures, full S.S
run 10  "$NC_B10"  --filling 0.5 --jperp 1 --szsz 1
run 100 "$NC_B100" --filling 0.5 --jperp 1 --szsz 1

# The two coupling decompositions, at the validated reference point only
run 10 "$NC_B10" --filling 0.5 --jperp 1 --szsz 0     # spin-flip only
run 10 "$NC_B10" --filling 0.5 --jperp 0 --szsz 1     # Sz.Sz only

# Away from half filling, both temperatures, full S.S -- needs the calibrated mu
if [ -n "$MU_B10_N075" ]; then
  run 10  "$NC_B10"  --filling 0.75 --mu "$MU_B10_N075"  --jperp 1 --szsz 1
else
  echo "SKIPPING beta=10 n=0.75: set MU_B10_N075 (see calibrate_mu.py)" >&2
fi
if [ -n "$MU_B100_N075" ]; then
  run 100 "$NC_B100" --filling 0.75 --mu "$MU_B100_N075" --jperp 1 --szsz 1
else
  echo "SKIPPING beta=100 n=0.75: set MU_B100_N075 (see calibrate_mu.py)" >&2
fi

# ---------------------------------------------------------------------------------------
# Extended grid (uncomment as needed)
# ---------------------------------------------------------------------------------------
# The coupling decompositions at the other three (beta, filling) points:
#run 100 "$NC_B100" --filling 0.5 --jperp 1 --szsz 0
#run 100 "$NC_B100" --filling 0.5 --jperp 0 --szsz 1
#
# CTHYB fully stochastic (no Lang-Firsov) -- the independent internal cross-check. The sign
# degrades badly with J: earlier runs recorded ~0.6 at J = 0.5 and ~0.3 at J = 1, so give it
# more cycles and expect noise.
#if [ "$SOLVER" = cthyb ]; then
#  run 10 $((NC_B10 * 4)) --filling 0.5 --jperp 1 --szsz 1 --lang_firsov False
#fi
