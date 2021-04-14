#from mpi4py import MPI
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
print(rank, a.larray.dtype, a.larray.device, a.larray.layout)
# cache = np.from_file('')
cache = np.random.rand(1000, 1000*int(sys.argv[1]))
a.larray = torch.from_numpy(cache).cuda(rank)
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

x = np.fromfile('rbc_conf_3264_m0.004_0.03_000290', dtype='>f8')
x = x.reshape(4,3,3,2,64,32,32,32)
x = x.transpose((4,5,6,7,0,2,1,3))
x = x[..., 0] + x[..., 1] * 1j
print(x.shape)

nt_per_gpu = 64 // comm.size 
f = open("rbc_conf_3264_m0.004_0.03_000290", "rb")
for i in range(4*3*3*2):
  f.seek(32**3 * nt_per_gpu * i, os.SEEK_SET)
  cache = np.fromfile(f, dtype='>f8', size=32**3 * nt_per_gpu)
