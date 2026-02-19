#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com 
#SBATCH --mail-type=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=ccq
#SBATCH --constraint=icelake
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
MODULES="modules/2.4 gcc flexiblas openmpi cmake ccache gmp fftw nfft hdf5/mpi boost python/3.12 python-mpi/3.12 intel-oneapi-mkl llvm/19 eigen mpfr"
module purge
module load ${MODULES}
#source /mnt/home/ahardy/ccq-software-build/triqs/3_unst_nix2.3_llvm/installation/share/triqs/triqsvars.sh
source /mnt/home/ahardy/ccq-software-build/triqs/3.3.x/.triqs_3/bin/activate
# source /mnt/home/ahardy/ccq-software-build/triqs/3_development/installation/share/triqs/triqsvars.sh
source /mnt/home/ahardy/ccq-software-build/triqs/3.3.x/installation/share/triqs/triqsvars.sh

##################### run your code here #####################
#../CTHYB/benchmark/
#mpirun python spin_spin.py --J 1.0 --U 4.0 --i1 0.0 --i2 0.0 --i3 0.0 --i4 0.0  --i5 0.0 --beta 10.0
#mpirun python ctseg_spin_spin.py --J 1.0 --U 4.0 --i1 0.0 --i2 0.0 --i3 0.0 --i4 0.0  --i5 0.0 --beta 10.0
#mpirun python spin_spin.py --J 1.0 --U 4.0 --i1 0.0 --i2 1.0 --i3 1.0 --i4 1.0  --i5 1.0 --beta 10.0
#mpirun python ctseg_spin_spin.py --J 1.0 --U 4.0 --i1 0.0 --i2 1.0 --i3 1.0 --i4 1.0  --i5 1.0 --beta 10.0
#mpirun python spin_spin.py --J 1.0 --U 4.0 --i1 1.0 --i2 0.0 --i3 0.0 --i4 0.0  --i5 0.0 --beta 10.0
mpirun python ctint_old_spin_spin.py --J 1.0 --U 4.0 --i1 1.0 --i2 0.0 --i3 0.0 --i4 0.0  --i5 0.0 --beta 10.0
