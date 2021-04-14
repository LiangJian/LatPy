from mpi4py import MPI
import timeit
import numpy as np
import time
from print0 import print0

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ngpu = comm.Get_size()
Nd = 32**3*64*4*9*2  # number of double
print0(rank, 'gauge size:', Nd*8, "Byte", fflush=True)

x = None
t1 = time.time()
if rank == 0:
  # x = np.fromfile('rbc_conf_3264_m0.004_0.03_000290', dtype='>f8')
  x = np.fromfile('rbc_conf_3264_m0.004_0.03_000290')
  print0(rank, x.shape, x[-1])
t2 = time.time()
print0(rank, "Using numpy on single core, cost", (t2-t1), 's', " BW is", Nd*8/1024/2014/2014/(t2-t1), "GB/s.", fflush=True)

if rank == 0:
  x = x.reshape(4,3,3,2,64,32,32,32)
  print("the following check should be:", flush=True)
  for i in range(ngpu):
    print(i, x[0,0,0,0,(64//ngpu*i):(64//ngpu*(i+1)),0,0,0][0], x[-1,-1,-1,-1,(64//ngpu*i):(64//ngpu*(i+1)),-1,-1,-1][-1], flush=True)
comm.barrier()

# x = x.reshape(4,3,3,2,64,32,32,32)
# x = x.transpose((4,5,6,7,0,2,1,3))
# x = x[..., 0] + x[..., 1] * 1j
sizes = (4,3,3,2,64,32,32,32)

subsizes = (4,3,3,2,64//ngpu,32,32,32)
starts = (0,0,0,0,64//ngpu*rank,0,0,0)
gauge_subarray = MPI.DOUBLE.Create_subarray(sizes, subsizes, starts)
print(rank, 'gauge_subarray:', gauge_subarray.lb, gauge_subarray.ub, gauge_subarray.size, gauge_subarray.extent, flush=True)

gauge_subarray.Commit()
fh = MPI.File.Open(comm, "rbc_conf_3264_m0.004_0.03_000290", MPI.MODE_RDONLY)
fh.Set_view(0, filetype=gauge_subarray)

buffer = np.empty(shape=subsizes)
t1 = time.time()
fh.Read_all(buffer)
comm.barrier()
t2 = time.time()
print0(rank, "Using MPI subarray on 4 cores, cost", (t2-t1), 's', " BW is", Nd*8/1024/2014/2014/(t2-t1), "GB/s.", fflush=True)
gauge_subarray.Free()
fh.Close()

print(rank, buffer[0,0,0,0,0,0,0,0], buffer[-1,-1,-1,-1,-1,-1,-1,-1], flush=True)
comm.barrier()

###################

distribs = [MPI.DISTRIBUTE_NONE] * 8
distribs[5] = MPI.DISTRIBUTE_BLOCK
dargs = [MPI.DISTRIBUTE_DFLT_DARG] * 8
psizes = [1,1,1,1,ngpu,1,1,1]
gauge_darray = MPI.DOUBLE.Create_darray(ngpu, rank, sizes, distribs, dargs, psizes)
print(rank, 'gauge_darray:', gauge_darray.lb, gauge_darray.ub, gauge_darray.size, gauge_darray.extent, flush=True)

gauge_darray.Commit()
fh = MPI.File.Open(comm, "rbc_conf_3264_m0.004_0.03_000290", MPI.MODE_RDONLY)
fh.Set_view(0, filetype=gauge_darray)

buffer = np.empty(shape=subsizes)
t1 = time.time()
fh.Read_all(buffer)
comm.barrier()
t2 = time.time()
print0(rank, "Using MPI darray on 4 cores, cost", (t2-t1), 's', " BW is", Nd*8/1024/2014/2014/(t2-t1), "GB/s.", fflush=True)
gauge_darray.Free()
fh.Close()

print(rank, buffer[0,0,0,0,0,0,0,0], buffer[-1,-1,-1,-1,-1,-1,-1,-1], flush=True)
