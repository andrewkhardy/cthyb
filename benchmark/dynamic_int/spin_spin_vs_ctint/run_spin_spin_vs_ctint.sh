#!/bin/bash
# Single-orbital spin-spin benchmark: CTHYB and CTSEG against CTINT, same bath and couplings.
# Submits one SLURM job per (solver, J, case). Model and factor-of-2 conventions are in
# spin_spin_common.py (CTHYB = CTSEG inputs; CTINT = half, since its action has no 1/2).
#
# Usage:
#   ./run_spin_spin_vs_ctint.sh                  # all three solvers
#   ./run_spin_spin_vs_ctint.sh cthyb ctseg      # a subset
#   DRY_RUN=1 ./run_spin_spin_vs_ctint.sh        # print the job scripts, submit nothing
#
# When the jobs are done:
#   python plot_spin_spin_vs_ctint.py --J 1.0 --data_dir $OUT_DIR --out spin_spin_vs_ctint_J-1.png
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR=/mnt/home/ahardy/ceph/CTHYB_Data/spin_spin_vs_ctint
LOG_DIR=/mnt/home/ahardy/ceph/SLURMOutputs
MAIL=andrewkhardy@protonmail.com
DRY_RUN=${DRY_RUN:-0}

U=4.0
J_VALUES=(0.5 1.0)
CASES=("1 1" "1 0" "0 1")   # "jperp szsz": full S.S, spin-flip only, Sz.Sz only

N_CYCLES_CTHYB=1000000
N_CYCLES_CTSEG=1000000
N_CYCLES_CTINT=5000000
CTHYB_LANG_FIRSOV=True      # False also sends Sz.Sz through the stochastic path (expect a sign problem)

SOLVERS=("$@")
[ ${#SOLVERS[@]} -eq 0 ] && SOLVERS=(cthyb ctseg ctint)

# Resources and environment per solver, as in the existing launchers:
#   cthyb: run_kanamori_dyn_spinflip.sh (this branch: triqs/multiorbital)
#   ctseg: run_benchmark.sh             (triqs/unstable)
#   ctint: run_compare.sh               (nix triqs/3_unst_nix2.4_llvm, icelake)
resources() {
  case "$1" in
    cthyb|ctseg) printf '%s\n' "#SBATCH --partition=ccq" "#SBATCH --ntasks-per-node=96" ;;
    ctint)       printf '%s\n' "#SBATCH --partition=ccq" "#SBATCH --constraint=icelake" "#SBATCH --ntasks-per-node=64" ;;
  esac
}

environment() {
  case "$1" in
    cthyb) printf '%s\n' "module purge" "module use /mnt/home/wentzell/opt/modules" \
                         "module load devenv9/clang-py3-mkl llvm/20" "module load triqs/multiorbital" \
                         "LAUNCH=\"mpirun -n 96\"" ;;
    ctseg) printf '%s\n' "module purge" "module use /mnt/home/wentzell/opt/modules" \
                         "module load devenv9/clang-py3-mkl llvm/20" "module load triqs/unstable" \
                         "LAUNCH=\"mpirun -n 96\"" ;;
    ctint) printf '%s\n' "module purge" \
                         "module load modules/2.4 gcc flexiblas openmpi cmake ccache gmp fftw nfft hdf5/mpi boost python/3.12 python-mpi/3.12 intel-oneapi-mkl llvm/19 eigen mpfr" \
                         "module load triqs/3_unst_nix2.4_llvm" \
                         "LAUNCH=srun" ;;
  esac
}

solver_args() {
  case "$1" in
    cthyb) echo "--n_cycles $N_CYCLES_CTHYB --lang_firsov $CTHYB_LANG_FIRSOV" ;;
    ctseg) echo "--n_cycles $N_CYCLES_CTSEG" ;;
    ctint) echo "--n_cycles $N_CYCLES_CTINT" ;;
  esac
}

submit() {
  local solver=$1 J=$2 jperp=$3 szsz=$4
  local job
  job=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=ss_${solver}_J${J}_jperp${jperp}_szsz${szsz}
#SBATCH --mail-user=${MAIL}
#SBATCH --mail-type=END,FAIL
#SBATCH --output=${LOG_DIR}/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
$(resources "$solver")
$(environment "$solver")
cd "${SCRIPT_DIR}"
\$LAUNCH python run_${solver}.py --J ${J} --U ${U} --jperp ${jperp} --szsz ${szsz} --out_dir ${OUT_DIR} $(solver_args "$solver")
EOF
)
  if [ "$DRY_RUN" = 1 ]; then
    printf '%s\n\n' "$job"
  else
    printf '%s\n' "$job" | sbatch
  fi
}

for solver in "${SOLVERS[@]}"; do
  case "$solver" in cthyb|ctseg|ctint) ;; *) echo "Unknown solver '$solver' (expected cthyb, ctseg, ctint)" >&2; exit 1 ;; esac
done
[ "$DRY_RUN" = 1 ] || mkdir -p "$OUT_DIR"

for solver in "${SOLVERS[@]}"; do
  for J in "${J_VALUES[@]}"; do
    for case_switches in "${CASES[@]}"; do
      read -r jperp szsz <<< "$case_switches"
      submit "$solver" "$J" "$jperp" "$szsz"
    done
  done
done
