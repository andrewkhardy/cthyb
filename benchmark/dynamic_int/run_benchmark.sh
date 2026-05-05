#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com 
#SBATCH --mail-type=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=ccq
#SBATCH --constraint=rome
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
MODULES="modules/2.4 gcc flexiblas openmpi cmake ccache gmp fftw nfft hdf5/mpi boost python/3.12 python-mpi/3.12 intel-oneapi-mkl llvm/19 eigen mpfr"
module purge
module load ${MODULES}
#module load triqs/3_unst_nix2.4_llvm
module load triqs/unstable
##################### run your code here #####################
#../CTHYB/benchmark/
mpirun -n 120 python spin_spin.py --J 1.0 --U 4.0 --i1 1.0 --i2 0.0 --i3 0.0 --i4 0.0  --i5 0.0 --beta 10.0 --n_cycles 100000 --measure_O_tau 50
#mpirun -n 120 python ctseg_spin_spin.py --J 1.0 --U 4.0  --i1 1.0 --i2 0.0 --i3 0.0 --i4 0.0  --i5 0.0 --beta 10.0
#mpirun -n 120 python ctint_spin_spin.py --J 1.0 --U 4.0  --i1 1.0 --i2 0.0 --i3 0.0 --i4 0.0  --i5 0.0 --beta 10.0
