def print0(rank, *args):
  if rank == 0:
    for arg in args:
      print(arg, end=' ')
    print('')
