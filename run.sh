#PBS -N test
##PBS -j oe
#PBS -e error.txt
#PBS -o output.txt
#PBS -l walltime=0:30:00
#PBS -q gpu
#PBS -l nodes=gpu01:ppn=4+gpu02:ppn=4
##PBS -l nodes=gpu01:ppn=4
##PBS -l nodes=1:ppn=12
#PBS -V

#conda activate py37
#module load myanaconda3

path=$PBS_O_WORKDIR
cd $path

# outfile=test1.out
# 
# date > $outfile
# echo $HOSTNAME >> $outfile

#mpirun -n 1 python test4.py 1 > test4.n1.out 2>&1 
#mpirun -n 2 python test4.py 2 > test4.n2.out 2>&1 
#mpirun -n 4 python test4.py 4 > test4.n4.out 2>&1 
mpirun -n 8 python test4.py 8 > test4.n8.out 2>&1 
