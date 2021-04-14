def print0(rank, *args, fflush=False):
  if rank == 0:
    for arg in args:
      print(arg, end=' ')
    print('', flush=fflush)
