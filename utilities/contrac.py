#!/beegfs/home/liuming/software/install/python/bin/python3
import numpy as np
import cupy as cp
import os
import fileinput
from gamma import *
from input_output import *

infile=fileinput.input()
for line in infile:
	tmp=line.split()
	if(tmp[0]=='Nt'):
		Nt=int(tmp[1])
	if(tmp[0]=='Nx'):
		Nx=int(tmp[1])
	if(tmp[0]=='conf_id'):
		conf_id=tmp[1]
	if(tmp[0]=='Nev'):
		Nev=int(tmp[1])
	if(tmp[0]=='peram_dir'):
		peram_dir=tmp[1]
	if(tmp[0]=='corr_dir'):
		corr_dir=tmp[1]

#peram_size=Nt*4*Nev*Nev*2
corr=cp.zeros((Nt), dtype=complex)
for t_source in range(Nt):
	f=open("%s/perams.%s.0.%i"%(peram_dir,conf_id,t_source),'rb')
	f.seek(8,os.SEEK_SET)
	peram=np.fromfile(f,dtype='>f8')
	peram=peram.byteswap(inplace=True)
	f.close()
	for d_source in range(1,4):
		f=open("%s/perams.%s.%i.%i"%(peram_dir,conf_id,d_source,t_source),'rb')
		f.seek(8,os.SEEK_SET)
		temp=np.fromfile(f,dtype='>f8')
		temp=temp.byteswap(inplace=True)
		peram=cp.append(peram, temp)
		f.close()
	peram=peram.reshape(4,Nt,Nev,4,Nev,2) #d_source, t_sink, ev_source, d_sink, ev_sink, complex
	peram=peram.transpose(1,4,2,3,0,5) #t_sink, ev_sink, ev_souce, d_sink, d_source, complex
	peram=peram[...,0] + peram[...,1]*1j

#	peram_anti = (g5*peram*g5)^dagger
	peram_anti=peram
	peram_anti[:,:,:,0:2,2:4]=-1.0*peram[:,:,:,0:2,2:4]
	peram_anti[:,:,:,2:4,0:2]=-1.0*peram[:,:,:,2:4,0:2]
	peram_anti=peram_anti.conj().transpose(0,2,1,4,3)

	corr_temp = cp.einsum('ijklm,ikjml->i',peram, peram_anti)
	corr = corr + cp.roll(corr_temp, -t_source)
corr = cp.asnumpy(corr)
corr = corr.reshape(1,Nt)
#print(corr.shape[0])
#	np.savetxt("%s/corr_test"%corr_dir, corr)
write_data_ascii(corr, Nt, Nx, "%s/corr_test"%corr_dir)
