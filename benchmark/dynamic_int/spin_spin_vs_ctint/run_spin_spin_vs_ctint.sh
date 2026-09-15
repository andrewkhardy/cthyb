#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com
#SBATCH --mail-type=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=ccq
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
module load modules/2.5-beta1
module load triqs/multiorbital     # cthyb
#module load triqs/unstable        # ctseg
# ctint (stack from run_compare.sh):
#module purge
#module load modules/2.4 gcc flexiblas openmpi cmake ccache gmp fftw nfft hdf5/mpi boost python/3.12 python-mpi/3.12 intel-oneapi-mkl llvm/19 eigen mpfr
#module load triqs/3_unst_nix2.4_llvm
##################### run your code here #####################
# Single-orbital spin-spin benchmark against CTINT; model and factors of 2 in spin_spin_common.py.
#   --jperp 1 --szsz 1 : full S.S     --jperp 1 --szsz 0 : spin-flip only     --jperp 0 --szsz 1 : Sz.Sz only
# Output: /mnt/home/ahardy/ceph/CTHYB_Data/spin_spin_vs_ctint/
# Plot:   python plot_spin_spin_vs_ctint.py --J 1.0

mpirun -n 96 python run_cthyb.py --J 1.0 --jperp 1 --szsz 1 --n_cycles 1000000
#mpirun -n 96 python run_cthyb.py --J 1.0 --jperp 1 --szsz 0 --n_cycles 1000000
#mpirun -n 96 python run_cthyb.py --J 1.0 --jperp 0 --szsz 1 --n_cycles 1000000

# CTHYB fully stochastic (no Lang-Firsov). Short local probe: sign ~0.6 at J=0.5, ~0.3 at J=1
#mpirun -n 96 python run_cthyb.py --J 0.5 --jperp 1 --szsz 1 --n_cycles 2000000 --lang_firsov False
#mpirun -n 96 python run_cthyb.py --J 1.0 --jperp 1 --szsz 1 --n_cycles 4000000 --lang_firsov False

#mpirun -n 96 python run_ctseg.py --J 1.0 --jperp 1 --szsz 1 --n_cycles 1000000
#mpirun -n 96 python run_ctseg.py --J 1.0 --jperp 1 --szsz 0 --n_cycles 1000000
#mpirun -n 96 python run_ctseg.py --J 1.0 --jperp 0 --szsz 1 --n_cycles 1000000

#srun python run_ctint.py --J 1.0 --jperp 1 --szsz 1 --n_cycles 5000000
#srun python run_ctint.py --J 1.0 --jperp 1 --szsz 0 --n_cycles 5000000
#srun python run_ctint.py --J 1.0 --jperp 0 --szsz 1 --n_cycles 5000000
