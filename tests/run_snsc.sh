#!/bin/bash
#SBATCH --partition=gpu8
#SBATCH --job-name=test_heat
#SBATCH --output=test_heat.log
#SBATCH --error=test_heat.log
#SBATCH --nodes=1
#SBATCH -n 4
# #SBATCH --exclusive
#SBATCH --cpus-per-task=4
# #SBATCH --nodelist= 
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:4

# GPU8  36 Phyisical cores
# DGX2  48 Phyisical cores

export export OMP_NUM_THREADS=4
export OMPI_MCA_btl=^vader,tcp,openib

# mpirun -np 4 python test1.py > test1_snsc.out 2>&1
# mpirun -np 4 python test2.py > test2_snsc.out 2>&1
# mpirun -np 4 python test3.py > test3_snsc.out 2>&1

mpirun -n 4 python test_MPI-IO.py > test_MPI-IO_snsc.out 2>&1
