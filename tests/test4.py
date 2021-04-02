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
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#ngpu = comm.Get_size()

t = torch.cuda.get_device_properties(0).total_memory / 1024. / 1024. /1024. 
r = torch.cuda.memory_reserved(0) / 1024. / 1024.
a = torch.cuda.memory_allocated(0) / 1024. / 1024.
print0(rank, "Total, reserved, allocated GPU memory =", t, "GB,", r, "MB,", a, "MB,")
torch.cuda.empty_cache()
t = torch.cuda.get_device_properties(0).total_memory / 1024. / 1024. /1024. 
r = torch.cuda.memory_reserved(0) / 1024. / 1024.
a = torch.cuda.memory_allocated(0) / 1024. / 1024.
print0(rank, "Total, reserved, allocated GPU memory =", t, "GB,", r, "MB,", a, "MB,")

N = 50

size = 5000 * int(sys.argv[1])

a = ht.random.rand(size, size, dtype=ht.float64, split=0, device='gpu')
b = ht.random.rand(size, size, dtype=ht.float64, split=1, device='gpu')
t1 = time.time()
for i in range(N):
  c = a @ b
t2 = time.time()
print0(rank, "Using HeAT on gpus, matrix size {:d}, time taken is ".format(size), (t2-t1)/100., "s.")


