#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=ccq
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#
# Two-patch valence-bond dimer with a retarded real-space spin-spin interaction.
# Model shared by every driver: model.py
#
#   usage:  sbatch run_vb_dimer.sh cthyb     # the QMC grid
#           sbatch run_vb_dimer.sh ed        # the ED references -- a cluster job, see below
#           python check_rotation.py         # free symbolic pre-flight, no MC
#
# The ED here is NOT cheap, unlike kanamori_phonon's. The block dimension is
# 70 x (n_ph+1)^(3 x rank), so the rank-1 J_ED point at --n_ph 3 is a dense eigh on
# ~4500-dimensional blocks, done twice when --n_ph_check is on. Submit it; do not run it
# on a laptop. (--n_ph 2 --n_ph_check 0 is ~16 s and is the only sensible local smoke test.)
#
# This is the only benchmark here whose dynamical vertices are OFF-DIAGONAL in the solver's
# working basis: U and the spin-spin coupling are local on the rotated cluster sites, so in
# the patch basis they become c^dag_K c_K' bilinears with K != K'. That is what exercises
# the general 4-index stochastic expansion.
#
# TWO POINTS, and they answer different questions
# -----------------------------------------------
# J_DCA  (J_intra = 0, J_inter = 0.5) is the physically interesting VBDMFT coupling, but it
#   has NO exact reference and cannot have one. Integrating out harmonic bosons always
#   yields a kernel -(positive semidefinite) x |Q|, so an ED exists only when -J is psd,
#   i.e. J_intra <= -|J_inter|. Here -J has eigenvalues +-0.5. CTHYB samples it fine (it is
#   a perfectly well-defined action); ED simply has no Hamiltonian to diagonalize, and
#   run_ed.py refuses it by design rather than silently building a non-Hermitian H.
#   Cross-check for this point: lang_firsov True vs False, two different routes to the same
#   physics.
#
# J_ED   (J_intra = -J_inter = -0.5) is the nearest ED-representable point: -J is rank 1,
#   one boson triplet coupled to (S_1 - S_2)/sqrt(2). This is how an antiferromagnetic
#   S_1.S_2 physically arises -- -g^2 (S_1-S_2)^2 contains +2g^2 S_1.S_2 -- so it is a
#   sensible model in its own right, not a contrivance. Here ED is exact and CTHYB is
#   measured against it.
#
# Plot with:  python plot_vb_dimer.py

set -euo pipefail
SOLVER="${1:-}"
case "$SOLVER" in
  cthyb|ed) ;;
  *) echo "usage: sbatch $0 cthyb|ed" >&2; exit 2 ;;
esac

module load modules/2.5-beta1
module load triqs/multiorbital

NRANKS=96
OUT=/mnt/home/ahardy/ceph/CTHYB_Data/vb_dimer

COMMON="--t 0.25 --tp 0.0 --U 2.0 --omega_0 1.0"
# ED-representable: needs the discrete bath too, since the DCA coarse-grained
# hybridization is continuous and has no finite Hamiltonian representation.
J_ED="--J_intra -0.5 --J_inter 0.5 --bath discrete --V 0.5 --eps_bath 0.0"
# The physical DCA point, on the real coarse-grained bath.
J_DCA="--J_intra 0.0 --J_inter 0.5 --bath dca"

# ---------------------------------------------------------------------------------------
# Chemical potentials
# ---------------------------------------------------------------------------------------
# A pure spin-spin retarded interaction carries no static charge shift, so half filling is
# mu = U/2 exactly (model.py's default), same argument as the single-orbital spin-spin
# benchmark. Away from half filling:
#   ED point : python calibrate_mu.py --target_n 0.75 $J_ED   (exact ED probe; submit it,
#              each solve is a dense eigh -- see the note on ED cost in the header)
#   DCA point: no ED available, so a short CTHYB probe is the only option, and it is needed
#              at half filling too because mu = U/2 is not exact there:
#                mpirun -n 16 python calibrate_mu.py --target_n 0.5  $J_DCA
#                mpirun -n 16 python calibrate_mu.py --target_n 0.75 $J_DCA
#
# Calibrated 2026-09-21, ED probe at the default --n_ph 2; beta = 10 only so far.
#   python calibrate_mu.py --beta 10 --target_n 0.75 $J_ED
#
# The probe reports deviation 0, but that is zero to its own resolution, not to machine
# precision: run_ed.py prints <n_a> rounded to 5 decimals and calibrate_mu.py bisects on
# that printed line, so n cannot be resolved below ~1e-5 however tight --tol is set. Still
# two orders tighter than the stochastic CTSEG probes the single-orbital benchmarks use.
#
# The probe also truncates at n_ph = 2 while the reference runs use n_ph = 3, so the filling
# actually realized at this mu may differ in the last digits. That is harmless here, because
# both ED and CTHYB take the pinned mu and therefore solve the identical Hamiltonian -- only
# the "n = 0.75" label is nominal, never the comparison.
MU_B10_N075=2.404877    # $J_ED,  n = 0.75  -> n = 0.750000
MU_B100_N075=""         # $J_ED,  n = 0.75  -- beta = 100 ED probe not yet run
MU_DCA_B10_N05=""       # $J_DCA, n = 0.50 (not U/2: the DCA bath breaks ph symmetry)
MU_DCA_B100_N05=""

# ---------------------------------------------------------------------------------------
# ED references -- serial, but memory and CPU heavy: submit it (see the header)
# ---------------------------------------------------------------------------------------
if [ "$SOLVER" = ed ]; then
  for BETA in 10.0 100.0; do
    python run_ed.py $COMMON $J_ED --beta $BETA --n_ph 3 --n_ph_check 1 --out_dir "$OUT"
  done
  echo "ED references written to $OUT"
  echo "NOTE: no ED for the J_DCA point -- see the header for why."
  exit 0
fi

# ---------------------------------------------------------------------------------------
# Statistics and wall-clock
# ---------------------------------------------------------------------------------------
# 45 min budget. Expect the worst sign of the whole benchmark set here: the vertices are
# off-diagonal in the working basis, so none of them is eligible for Lang-Firsov and the
# whole coupling is sampled stochastically. Earlier runs of this model recorded 48 vertices
# for J_intra=0/J_inter=0.5. Budget accordingly and read the reported sign before trusting
# any curve.
MAX_TIME=2700
NC_B10=500000
NC_B100=50000

run () {  # run <beta> <n_cycles> <model args...>
  local beta="$1"; local ncyc="$2"; shift 2
  echo "=== cthyb  beta=$beta  n_cycles=$ncyc  $* ==="
  mpirun -n "$NRANKS" python run_cthyb.py $COMMON --beta "$beta" --n_cycles "$ncyc" \
      --n_warmup_cycles $((ncyc / 20)) --max_time "$MAX_TIME" --out_dir "$OUT" "$@"
}

# ------------------------------------------------- the ED-comparable point, both betas
run 10  "$NC_B10"  $J_ED
run 100 "$NC_B100" $J_ED

# ------------------------------------------------- the physical DCA point, both betas
# No ED here; the lang_firsov True/False pair below is the cross-check.
# Unlike $J_ED, this point needs mu calibrated even AT HALF FILLING: the coarse-grained
# hybridization is not particle-hole symmetric (model.half_filling_is_exact() is False,
# patch DOS variance ratio 1.174), so mu = U/2 lands at an unknown filling. Verified: a
# short probe at mu = U/2 came out well away from n = 0.5.
if [ -n "$MU_DCA_B10_N05" ]; then
  run 10  "$NC_B10"  $J_DCA --mu "$MU_DCA_B10_N05"
else
  echo "SKIPPING beta=10 DCA half filling: set MU_DCA_B10_N05 (see calibrate_mu.py)" >&2
fi
if [ -n "$MU_DCA_B100_N05" ]; then
  run 100 "$NC_B100" $J_DCA --mu "$MU_DCA_B100_N05"
else
  echo "SKIPPING beta=100 DCA half filling: set MU_DCA_B100_N05 (see calibrate_mu.py)" >&2
fi

# n = 0.75, once calibrated
if [ -n "$MU_B10_N075" ]; then
  run 10 "$NC_B10" $J_ED --mu "$MU_B10_N075"
else
  echo "SKIPPING beta=10 n=0.75: set MU_B10_N075 (see calibrate_mu.py)" >&2
fi
if [ -n "$MU_B100_N075" ]; then
  run 100 "$NC_B100" $J_ED --mu "$MU_B100_N075"
else
  echo "SKIPPING beta=100 n=0.75: set MU_B100_N075 (see calibrate_mu.py)" >&2
fi

# --------------------------------------------------------------- the lf True/False pair
# For the DCA point this IS the benchmark, since no ED exists. The two runs must use the
# same mu to be comparable: run_cthyb.py does NOT probe for mu (it says so explicitly, and
# deliberately -- a probe would be route dependent), so both take the calibrated value
# above, which calibrate_mu.py always measures with lang_firsov=True for exactly this reason.
run 10 $((NC_B10 * 2)) $J_DCA ${MU_DCA_B10_N05:+--mu "$MU_DCA_B10_N05"} --lang_firsov False

# ---------------------------------------------------------------- extended (uncomment)
#run 100 $((NC_B100 * 2)) $J_DCA --lang_firsov False
#
# NOT worth running: the lf True/False pair is VACUOUS at $J_ED. That point has zero
# density-density vertices (check_rotation.py: "0 are density-density" at J_intra = -0.5,
# J_inter = 0.5, against 16 at the DCA point), because -J is rank 1 on (1,-1)/sqrt(2) so
# every monomial is patch-off-diagonal. With nothing Lang-Firsov-eligible, lf=True and
# lf=False build the identical stochastic catalog and the two runs differ only by seed.
#run 10  $((NC_B10 * 2))  $J_ED  --lang_firsov False
