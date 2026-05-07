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
# MODULES="modules/2.4 gcc flexiblas openmpi cmake ccache gmp fftw nfft hdf5/mpi boost python/3.12 python-mpi/3.12 intel-oneapi-mkl llvm/19 eigen mpfr"
# module purge
# module load ${MODULES}
MODULES="devenv9/clang-py3-mkl llvm/20"
module purge
module use /mnt/home/wentzell/opt/modules
module load ${MODULES}
module load triqs/unstable
##################### run your code here #####################
#../CTHYB/benchmark/
#mpirun -n 120 python holstein.py --L -0.01 --U 4.0  --beta 10.0 --n_cycles 100000 --measure_O_tau 150
mpirun -n 120 python ctseg_holstein.py --L -0.0 --U 4.0   --beta 10.0 --n_cycles 100000 
#mpirun -n 120 python ctint_holstein.py --J 1.0 --U 4.0  --i1 1.0 --i2 0.0 --i3 0.0 --i4 0.0  --i5 0.0 --beta 10.0
