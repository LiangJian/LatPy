import heat as ht
import timeit
import numpy as np
import cupy as cp
import time
from print0 import print0
from gamma import *
from gamma_matrix import *
from opt_einsum import contract

rank = ht.get_comm().rank

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
	peram=peram.reshape(4,Nt,Nev,4,Nev,2) #d_source, t_sink, ev_source, d_sink, ev_sink, complex
	peram=peram.transpose(1,4,2,3,0,5) #t_sink, ev_sink, ev_souce, d_sink, d_source, complex
	peram=peram[...,0] + peram[...,1]*1j
	peram=peram[:,0:Nev_use,0:Nev_use,:,:]
	return peram


def read_eig(eig_dir, conf_id, Nt, Nx, Nev, Nev_use, t_source):
	f=open("%s/eigvecs_t%03d_%d"%(eig_dir,t_source,conf_id),'rb')
	eigv=np.fromfile(f,dtype='f8')
	f.close()
	eigv=eigv.reshape(Nev,Nx**3,3,2)
	eigv=eigv[...,0] + eigv[...,1]*1j
	eigv=eigv[0:Nev_use,...]
	return eigv


peram_dir = '/beegfs/group/lqcd/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/test_Nev600/7400'
eig_dir = '/beegfs/home/liuming/LapH/data/eigensystem/conf_32x96_b6.41_2295_2050/hyp_smear/Nev600/7400'
conf_id = 7400
Nev = 600
Nev_use = 300
Nx = 32
Nt = 96

cg5 = g2 @ g4 @ g5
cg5 = ht.array(cg5, device='gpu')
g5 = ht.array(g5, device='gpu')

#vv = []
#for t_source in range(0,Nt,1):
#	eigv = read_eig(eig_dir, conf_id, Nt, Nx, Nev, Nev_use, t_source)
#	vv.append(contract('ijk,ljm->ilkm',eigv.conj(), eigv))
#	print(t_source,end='\t', flush=True)
#vv = np.array(vv)
#vv = vv.reshape(Nt,Nev_use,Nev_use,3,3)
#np.save('vv_%d'%conf_id,vv)
vv = np.load(('vv_%d.npy'%conf_id))
print("vv done")  #t_source, ev_1, ev_2, c_1, c_2
print(vv.shape)
vv = ht.array(vv, device='gpu')

print(type(vv))
print(type(vv.array()))
exit(0)


for t_source in range(0,Nt,4):
	peram=readin_peram(peram_dir, conf_id, Nt, Nev, Nev_use, t_source)
	print("read peram done")
	print(peram.shape)
	peram = ht.array(peram, device='gpu')

	Qu = contract('ijklm,mn->ijkln', peram, cg5)  # t_sink, ev_sink, ev_souce, d_sink, d_source
	break
	Qd = contract('ij,klmin->klmjn', cg5, peram)  # t_sink, ev_sink, ev_souce, d_sink, d_source
	Qd = contract('ij,klmjn,no->klmio', g5, peram, g5)  # t_sink, ev_sink, ev_souce, d_sink, d_source
	Qd = Qd.conj()  # t_sink, ev_sink, ev_souce, d_sink, d_source
	print(Qu.shape, Qd.shape)
	break
	T = contract('ijklm, knop ->ijnlmop',Qu,vv[t_source,...])
	print(T.shape)
	R = contract('ijnlmop, iqnmr->ilropjq',T,Qd)
	print(R.shape)
	#R = contract('ijklm, knop, iqnmr->ilropjq',Qu,vv[t_source,...],Qd)
	break



