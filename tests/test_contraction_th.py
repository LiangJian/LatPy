import heat as ht
import torch as th
import timeit
import numpy as np
import cupy as cp
import time
from print0 import print0
from gamma import *
from gamma_matrix import *
from opt_einsum import contract

rank = ht.get_comm().rank
cuda = th.device('cuda')


def readin_peram(peram_dir, conf_id, Nt, Nev, Nev_use, t_source):
	f=open("%s/perams.%s.0.%i"%(peram_dir,conf_id,t_source),'rb')
	peram=np.fromfile(f,dtype='f8')
	f.close()
	for d_source in range(1,4):
		f=open("%s/perams.%s.%i.%i"%(peram_dir,conf_id,d_source,t_source),'rb')
		temp=np.fromfile(f,dtype='f8')
		peram=np.append(peram, temp)
		temp=None
		f.close()
	peram=peram.reshape(4,Nt,Nev,4,Nev,2)  # d_source, t_sink, ev_source, d_sink, ev_sink, complex
	peram=peram.transpose(1,4,2,3,0,5)  # t_sink, ev_sink, ev_souce, d_sink, d_source, complex
	peram=peram[...,0] + peram[...,1]*1j
	peram=peram[:,0:Nev_use,0:Nev_use,:,:]
	return peram


def read_eig(eig_dir, conf_id, Nt, Nx, Nev, Nev_use, t_source):
	f=open("%s/eigvecs_t%03d_%d"%(eig_dir,t_source,conf_id),'rb')
	eigv=np.fromfile(f,dtype='f8')
	f.close()
	eigv=eigv.reshape(Nev,Nx**3,3,2)  # ev, nx**3, color, complex
	eigv=eigv[...,0] + eigv[...,1]*1j  # ev, nx**3, color
	eigv=eigv[0:Nev_use,...]
	return eigv


peram_dir = '/beegfs/group/lqcd/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light/test_Nev600/7400'
eig_dir = '/beegfs/home/liuming/LapH/data/eigensystem/conf_32x96_b6.41_2295_2050/hyp_smear/Nev600/7400'
conf_id = 7400
Nev = 600
Nev_use = 150
Nx = 32
Nt = 96

cg5 = g2 @ g4 @ g5
cg5 = th.tensor(cg5, device='cuda')
g5 = th.tensor(g5, device='cuda')

#vv = []
#for t_source in range(0,Nt,1):
#	eigv = read_eig(eig_dir, conf_id, Nt, Nx, Nev, Nev_use, t_source)
#	vv.append(contract('ijk,ljm->ilkm',eigv.conj(), eigv))
#	print(t_source,end='\t', flush=True)
#vv = np.array(vv)
#vv = vv.reshape(Nt,Nev_use,Nev_use,3,3)
#np.save('vv_%d'%conf_id,vv)
vv = np.load(('vv_%d.npy'%conf_id))[:,:Nev_use,:Nev_use,...]
print("vv done")  #t_source, ev_1, ev_2, c_1, c_2
print(vv.shape)
vv = th.tensor(vv, device='cuda')


for t_source in range(0,Nt,4):
	st = time.time()
	peram=readin_peram(peram_dir, conf_id, Nt, Nev, Nev_use, t_source)
	print("read peram done")
	print(peram.shape)
	peram = th.tensor(peram, device='cuda') # t_sink, ev_sink, ev_souce, d_sink, d_source

	Qu = contract(peram, ('t_sink', 'ev_sink', 'ev_souce', 'd_sink', 'd_source'), cg5, ('d_source', 'd2_source'),
			('t_sink', 'ev_sink', 'ev_souce', 'd_sink', 'd2_source'))  # cg5 onto source
	
	Qd = contract(peram, ('t_sink', 'ev_sink', 'ev_souce', 'd_sink', 'd_source'), cg5, ('d2_sink', 'd_sink'),
			('t_sink', 'ev_sink', 'ev_souce', 'd2_sink', 'd_source'))  # cg5 onto sink
	
	Qd = contract(g5, ('d2_sink', 'd_sink'), peram.conj(), ('t_sink', 'ev_sink', 'ev_source', 'd_sink', 'd_source'), g5, ('d_source', 'd2_source'),
			('t_sink', 'ev_source', 'ev_sink', 'd2_source', 'd2_sink'))  # g5 hermitian 

	print(Qu.shape, Qd.shape)

	R = contract(Qu, ('t_sink', 'ev_sink', 'ev_souce', 'd_sink', 'd_source'), vv[t_source,...].conj(), ('ev_source', 'ev2_source', 'c_1', 'c_2'),
			Qd, ('t_sink', 'ev2_source', 'ev2_sink', 'd_source', 'd2_sink'), ('t_sink', 'ev_sink', 'ev2_sink', 'd_sink', 'd2_sink', 'c_1', 'c_2'))

	print(R.shape)
	ed = time.time()
	print('used', ed - st, 's')
	break

	TrR = contract(R,('t_sink', 'ev_sink', 'ev2_sink', 'd_sink', 'd_sink', 'c_1', 'c_2'),('t_sink', 'ev_sink', 'ev2_sink', 'c_1', 'c_2'))
	print(TrR.shape)  

	L = R.copy()
	# e_{a_1,b_1,c_1} e_{a_2,b_2,c_2} (a_2,b_1), (b_2,c_1)
	R[...,0,0] = contract(TrR,('t_sink', 'ev_sink', 'ev2_sink', 'a_2', 'b_1'),vv.conj(), ('t_sink','ev2_sink', 'ev3_sink', 'b_2', 'c_1'),
			peram,('t_sink', 'ev3_sink', 'ev_souce', 'd_sink', 'd_source'),('t_sink', 'ev_sink', 'ev_souce', 'd_sink', 'd_source'))

	#V = read_eig(eig_dir, conf_id, Nt, Nx, Nev, Nev_use, t_source)	
	#D = contract('ijnlmop, iqnmr->ilropjq',V,R,V)



