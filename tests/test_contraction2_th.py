import torch as th
import numpy as np
import time
from gamma import *
from opt_einsum import contract


def read_peram(peram_dir, conf_id, Nt, Nev, Nev_use, t_source):
	peram = []
	for d_source in range(4):
		peram.append(np.fromfile("%s/perams.%s.%i.%i"%(peram_dir,conf_id,d_source,t_source), dtype='c16'))
	peram = np.array(peram)	
	peram=peram.reshape(4,Nt,Nev,4,Nev)  # d_source, t_sink, ev_source, d_sink, ev_sink
	peram=peram.transpose(1,4,2,3,0)  # t_sink, ev_sink, ev_souce, d_sink, d_source
	peram=peram[:,:Nev_use,:Nev_use,:,:]
	return peram


def read_eig(eig_dir, conf_id, Nt, Nx, Nev, Nev_use, t_source):
	eigv=np.fromfile("%s/eigvecs_t%03d_%d"%(eig_dir,t_source,conf_id), dtype='c16')
	eigv=eigv.reshape(Nev,Nx**3,3)  # ev, nx**3, color
	eigv=eigv[:Nev_use,...]
	return eigv


def read_vvv(vvv_dir, conf_id, Nev, Nev_use, t_source):
	vvv=np.fromfile("%s/VVV.t%02d.conf%d"%(vvv_dir,t_source,conf_id), dtype='c16')
	vvv=vvv.reshape(Nev,Nev,Nev)
	vvv=vvv[:Nev_use,:Nev_use,:Nev_use]
	return vvv


peram_dir = '/beegfs/group/lqcd/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light/test_Nev600/7400/'
eig_dir = '/beegfs/home/liuming/LapH/data/eigensystem/conf_32x96_b6.41_2295_2050/hyp_smear/Nev600/7400/'
vvv_dir='/beegfs/home/liuming/LapH/data/VVV/conf_32x96_b6.41_2295_2050/7400/'
conf_id = 7400
Nev = 600
Nev_use = 200
Nx = 32
Nt = 96

cg5 = g2 @ g4 @ g5
# cg5 = th.tensor(cg5, device='cuda')
cg5 = th.tensor(cg5.astype('c8'), device='cuda')
Gamma = (g0 + g4)/2.
# Gamma = th.tensor(Gamma, device='cuda')
Gamma = th.tensor(Gamma.astype('c8'), device='cuda')

# vvv = []
# for t_source in range(0,Nt,1):
# 	eigv = read_eig(eig_dir, conf_id, Nt, Nx, Nev, Nev_use, t_source)  # i_ev, spatial, color
# 	eigv = th.tensor(eigv, device='cuda')
# 	temp = contract(eigv,('i_ev', 'x', 'i_c1'), eigv,('j_ev', 'x', 'i_c2'), eigv,('k_ev', 'x', 'i_c3'), ('i_ev', 'j_ev', 'k_ev', 'i_c1', 'i_c2', 'i_c3'))
# 	vvv.append(contract(temp, ('i_ev', 'j_ev', 'k_ev', 'i_c1', 'i_c2', 'i_c3'), epsilon, ('i_c1', 'i_c2', 'i_c3'), ('i_ev', 'j_ev', 'k_ev')))
# 	print(t_source,end='\t', flush=True)
# vvv = np.array(vvv)
# vvv = vvv.reshape(Nt,Nev_use,Nev_use,Nev_use)
# np.save('vvv_%d'%conf_id,vvv)
# vvv = np.load(('vvv_%d.npy'%conf_id))
# print("vvv done")  # t_source, ev_1, ev_2, ev_3 
# print(vvv.shape)
# vvv= th.tensor(vvv, device='cuda')

vvv = []
st = time.time()
for it in range(0, 96, 1):
	vvv.append(read_vvv(vvv_dir, conf_id, 300, Nev_use, it))
ed = time.time()
print('read vvv used %.4f'%(ed - st), 's.')
vvv = np.array(vvv)
print(vvv.shape)
st = time.time()
# vvv = th.tensor(vvv, device='cuda')
vvv = th.tensor(vvv.astype('c8'), device='cuda')  # to single
ed = time.time()
print('copy vvv to gpu used %.4f'%(ed - st), 's.')
print('GPU memory used %.4f'%(th.cuda.memory_allocated()/1024/1024/2014),'GB.')

res = []

dt_source = 4
dt_sink = 4

for t_source in range(0,Nt,dt_source):
	st = time.time()
	peram0=read_peram(peram_dir, conf_id, Nt, Nev, Nev_use, t_source)
	print(peram0.shape) # t_sink, ev_sink, ev_souce, d_sink, d_source
	ed = time.time()
	print('read peram with t0 = %d done, used %.4f'%(t_source, ed - st), 's.')

	st = time.time()
	for t_sink in range(0,Nt,dt_sink):
		peram = peram0[t_sink:t_sink+dt_sink,...]
		# P = th.tensor(peram, device='cuda')
		P = th.tensor(peram.astype('c8'), device='cuda')  # to single

		Qu = contract(P, ('t_sink', 'ev_sink', 'ev_souce', 'd_sink', 'd_source'), cg5, ('d_source', 'd2_source'),
				('t_sink', 'ev_sink', 'ev_souce', 'd_sink', 'd2_source'))  # cg5 onto source
		
		Qd = contract(P, ('t_sink', 'ev_sink', 'ev_souce', 'd_sink', 'd_source'), cg5, ('d2_sink', 'd_sink'),
				('t_sink', 'ev_sink', 'ev_souce', 'd2_sink', 'd_source'))  # cg5 onto sink
	
		V = contract(vvv[t_source, ...].conj(), ('j_ev', 'l_ev', 'n_ev'), Qu, ('t_sink', 'i_ev', 'j_ev', 'alpha_1', 'beta_2'),
				('t_sink', 'i_ev', 'l_ev', 'n_ev', 'alpha_1', 'beta_2'))

		T = contract(V, ('t_sink', 'i_ev', 'l_ev', 'n_ev', 'alpha_1', 'beta_2'), Qd, ('t_sink', 'k_ev', 'l_ev', 'alpha_1', 'beta_2'),
				('t_sink', 'i_ev', 'k_ev', 'n_ev'))

		V = contract(vvv[t_sink:t_sink+dt_sink,...], ('t_sink', 'i_ev', 'k_ev', 'm_ev'), T, ('t_sink', 'i_ev', 'k_ev', 'n_ev'), 
				('t_sink', 'm_ev', 'n_ev'))

		T = contract(Gamma, ('gamma_2', 'gamma_1'), P, ('t_sink', 'm_ev', 'n_ev', 'gamma_1', 'gamma_2'), 
				('t_sink', 'm_ev', 'n_ev'))

		# P1 = contract(V, ('t_sink', 'm_ev', 'n_ev'), T, ('t_sink', 'm_ev', 'n_ev'), ('t_sink'))  # there is a bug
		P1 = contract('ijk,ijk->i', V, T)
		res.append(np.array(P1.cpu()))

		#------------
		
		V = contract(vvv[t_source, ...].conj(), ('j_ev', 'l_ev', 'n_ev'), P, ('t_sink', 'i_ev', 'j_ev', 'alpha_1', 'gamma_2'),
				('t_sink', 'i_ev', 'l_ev', 'n_ev', 'alpha_1', 'gamma_2'))

		T = contract(V, ('t_sink', 'i_ev', 'l_ev', 'n_ev', 'alpha_1', 'gamma_2'), Qd, ('t_sink', 'k_ev', 'l_ev', 'alpha_1', 'beta_2'),
				('t_sink', 'i_ev', 'k_ev', 'n_ev', 'gamma_2', 'beta_2'))

		V = contract(T, ('t_sink', 'i_ev', 'k_ev', 'n_ev', 'gamma_2', 'beta_2'), Qu, ('t_sink', 'm_ev', 'n_ev', 'gamma_1', 'beta_2'),
				('t_sink', 'i_ev', 'k_ev', 'm_ev', 'gamma_2', 'gamma_1'))

		T = contract(Gamma, ('gamma_2', 'gamma_1'), V, ('t_sink', 'i_ev', 'k_ev', 'm_ev', 'gamma_2', 'gamma_1'), 
				('t_sink', 'i_ev', 'k_ev', 'm_ev'))

		# P2 = contract(vvv[t_sink:t_sink+dt_sink,...], ('t_sink', 'i_ev', 'k_ev', 'm_ev'), T, ('t_sink', 'i_ev', 'k_ev', 'm_ev'), ('t_sink'))
		P2 = contract('ijkl,ijkl->i', vvv[t_sink:t_sink+dt_sink,...], T)
		res.append(np.array(P2.cpu()))

	ed = time.time()
	print('contraction for peram with t0 = %d done, used %.4f'%(t_source, ed - st), 's.')
	print('max GPU memory used %.4f'%(th.cuda.max_memory_allocated()/1024/1024/2014),'GB.')

res = np.array(res)
res = res.reshape(Nt//dt_source,Nt//dt_sink,2,dt_sink)
res = res.transpose(2,0,1,3)
res = res.reshape(2,Nt//dt_source,Nt)

# for i in range(Nt):
# 	print(i, '%.8e'%res[0, 0, i].real, '%.8e'%res[0, 0, i].imag, '%.8e'%res[1, 0, i].real, '%.8e'%res[1, 0, i].imag)

res = res[0,...] + res[1,...]
np.save('res.%d'%conf_id, res)

for i in range(Nt):
	print(i, '%.8e'%res[0, i].real)

