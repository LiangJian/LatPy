import heat as ht
import timeit
import numpy as np
import cupy as cp
import time

rank = ht.get_comm().rank
N = 50

a = np.random.rand(10000, 10000)
b = np.random.rand(10000, 10000)
c = np.random.rand(10000, 10000)
# print(type(a[0,0]))
t1 = time.time()
for i in range(N):
  d = c + a * b
t2 = time.time()
print(rank, "Using numpy, time taken is ", t2-t1, "s.")

a = ht.random.rand(10000, 10000, dtype=ht.float64, split=0, device='cpu')
b = ht.random.rand(10000, 10000, dtype=ht.float64, split=0, device='cpu')
c = ht.random.rand(10000, 10000, dtype=ht.float64, split=0, device='cpu')
t1 = time.time()
for i in range(N):
  d = c + a * b
t2 = time.time()
print(rank, "Using HeAT on cpus, time taken is ", t2-t1, "s.")

a = cp.random.rand(10000, 2500)
b = cp.random.rand(10000, 2500)
c = cp.random.rand(10000, 2500)
t1 = time.time()
for i in range(N*100):
  d = c + a * b
t2 = time.time()
print(rank, "Uisng cupy, time taken is ", (t2-t1)*4/100., "s.")

a = ht.random.rand(10000, 10000, dtype=ht.float64, split=0, device='gpu')
b = ht.random.rand(10000, 10000, dtype=ht.float64, split=0, device='gpu')
c = ht.random.rand(10000, 10000, dtype=ht.float64, split=0, device='gpu')
t1 = time.time()
for i in range(N*100):
  d = c + a * b
t2 = time.time()
print(rank, "Using HeAT on gpus, time taken is ", (t2-t1)/100., "s.")


