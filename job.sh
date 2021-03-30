#PBS -N test
##PBS -j oe
#PBS -l walltime=0:10:00
#PBS -q gpu
#PBS -l nodes=gpu01:ppn=12
##PBS -l nodes=1:ppn=12
#PBS -V

conda activate py37
#module load myanaconda3

path=$PBS_O_WORKDIR
cd $path

# outfile=test1.out
# 
# date > $outfile
# echo $HOSTNAME >> $outfile

mpirun -n 4 python test1.py > test1.out 2>&1 
# mpirun -n 4 python test2.py > test2.out 2>&1 
# mpirun -n 4 python test3.py > test3.out 2>&1 
