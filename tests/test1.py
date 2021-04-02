import heat as ht
import timeit
import numpy as np
import cupy as cp
import time
from print0 import print0

rank = ht.get_comm().rank
N = 50

a = np.random.rand(10000, 10000)
b = np.random.rand(10000, 10000)
# print(type(a[0,0]))
t1 = time.time()
for i in range(N):
  c = a + b
t2 = time.time()
print0(rank, "Using numpy, time taken is ", t2-t1, "s.")

a = ht.random.rand(10000, 10000, dtype=ht.float64, split=0, device='cpu')
b = ht.random.rand(10000, 10000, dtype=ht.float64, split=0, device='cpu')
t1 = time.time()
for i in range(N):
  c = a + b
t2 = time.time()
print0(rank, "Using HeAT on cpus, time taken is ", t2-t1, "s.")

a = cp.random.rand(10000, 2500)
b = cp.random.rand(10000, 2500)
t1 = time.time()
for i in range(N*100):
  c = a + b
t2 = time.time()
print0(rank, "Uisng cupy, time taken is ", (t2-t1)*4/100., "s.")

a = ht.random.rand(10000, 10000, dtype=ht.float64, split=0, device='gpu')
b = ht.random.rand(10000, 10000, dtype=ht.float64, split=0, device='gpu')
t1 = time.time()
for i in range(N*100):
  c = a + b
t2 = time.time()
print0(rank, "Using HeAT on gpus, time taken is ", (t2-t1)/100., "s.")


