#from mpi4py import MPI
import timeit
import numpy as np
import cupy as cp
import time
from print0 import print0
import sys
import torch
import heat as ht

rank = ht.get_comm().rank
comm = ht.get_comm()
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#ngpu = comm.Get_size()

size = 1000 * int(sys.argv[1])

a = ht.random.rand(size, size, dtype=ht.float64, split=0, device='gpu')
ht.save(a, 'data.h5', 'DATA', mode='w')
comm.Barrier()

print0(rank, a.split)
print0(rank, type(a.larray), a.larray.shape)
print0(rank, a.gshape, a.lshape, a.split)
print(rank, a.larray.dtype, a.larray.device, a.larray.layout )
a.larray = torch.from_numpy(np.random.rand(1000, 1000*int(sys.argv[1]))).cuda(rank)
print0(rank, a.gshape, a.lshape, a.split)
print(rank, a.larray.dtype, a.larray.device, a.larray.layout )

t1 = time.time()
if rank == 0:
  b = ht.load('data.h5', dataset='DATA')
t2 = time.time()
comm.Barrier()
print0(rank, "the loading time is {:.2f} s for a {:.2f} MB matrix.".format((t2-t1),size*size/1024./1024.))

if rank == 0:
  print(rank, b.split)

