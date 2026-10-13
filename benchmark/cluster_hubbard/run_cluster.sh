#!/bin/bash
#SBATCH --mail-user=andrewkhardy@protonmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=ccq
#SBATCH --output=/mnt/home/ahardy/ceph/SLURMOutputs/%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=23:00:00
#
# One cluster DCA / CDMFT run on a node, of cluster_cthyb.py (CTHYB) or cluster_ctint.py (CTINT), whose defaults are
# those of the production runs. From this directory, e.g.
#   sbatch run_cluster.sh cluster_ctint.py --cluster 8A --scheme CDMFT
#   sbatch run_cluster.sh cluster_cthyb.py --cluster 2x2 --scheme DCA --basis sites --n_loops 1

module load modules/2.5-20261005
module load triqs/multiorbital
mpirun python "$1" --out_dir /mnt/home/ahardy/ceph/CTHYB_Data/cluster_hubbard/dlr_symm "${@:2}"
