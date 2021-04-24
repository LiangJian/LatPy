import numpy as np
import time
import numba

x = np.fromfile('./rbc_conf_2464_m0.00107_0.0850_000248', dtype='>f8')
#x = np.fromfile('/project/projectdirs/mp7/2464_005_04/rbc_conf_2464_m0.005_0.04_000495_hyp', dtype='>f8')
#x = np.fromfile('/project/projectdirs/mp7/2464_005_04_nohyp/rbc_conf_2464_m0.005_0.04_000495', dtype='>f8')
x = x.reshape(4,3,3,2,64,24,24,24)
x = x.transpose((4,5,6,7,0,2,1,3))
x = x[..., 0] + x[..., 1] * 1j
print(x.shape)
print('t z y x!')

# directions
xdir = {}
xdir['x'] = x[...,0,:,:]
xdir['y'] = x[...,1,:,:]
xdir['z'] = x[...,2,:,:]
xdir['t'] = x[...,3,:,:]

# location of coordinates indices
lOfI = {}
lOfI['x'] = 3
lOfI['y'] = 2
lOfI['z'] = 1
lOfI['t'] = 0

# arXiv:1509.04259


def plaq_general_1(xdir_, dir1_="x", dir2_="y"):
    tmp = np.einsum('...ij,...jk',xdir_[dir1_],np.roll(xdir[dir2_],-1,lOfI[dir1_]))
    tmp = np.einsum('...ij,...jk',tmp,np.roll(xdir_[dir1_],-1,lOfI[dir2_]).transpose((0,1,2,3,5,4)).conjugate())
    tmp = np.einsum('...ij,...jk',tmp,xdir_[dir2_].transpose((0,1,2,3,5,4)).conjugate())
    return tmp


def plaq_general_2(xdir_, dir1_="x", dir2_="y"):
    tmp = np.einsum('...ij,...jk',xdir_[dir2_],np.roll(np.roll(xdir_[dir1_],-1,lOfI[dir2_]),+1,lOfI[dir1_]).transpose((0,1,2,3,5,4)).conjugate())
    tmp = np.einsum('...ij,...jk',tmp,np.roll(xdir_[dir2_],+1,lOfI[dir1_]).transpose((0,1,2,3,5,4)).conjugate())
    tmp = np.einsum('...ij,...jk',tmp,np.roll(xdir_[dir1_],+1,lOfI[dir1_]))
    return tmp


def plaq_general_3(xdir_, dir1_="x", dir2_="y"):
    tmp = np.einsum('...ij,...jk',np.roll(xdir_[dir1_],+1,lOfI[dir1_]).transpose((0,1,2,3,5,4)).conjugate(),np.roll(np.roll(xdir_[dir2_],+1,lOfI[dir1_]),+1,lOfI[dir2_]).transpose((0,1,2,3,5,4)).conjugate())
    tmp = np.einsum('...ij,...jk',tmp,np.roll(np.roll(xdir_[dir1_],+1,lOfI[dir1_]),+1,lOfI[dir2_]))
    tmp = np.einsum('...ij,...jk',tmp,np.roll(xdir_[dir2_],+1,lOfI[dir2_]))
    return tmp


def plaq_general_4(xdir_, dir1_="x", dir2_="y"):
    tmp = np.einsum('...ij,...jk',np.roll(xdir_[dir2_],+1,lOfI[dir2_]).transpose((0,1,2,3,5,4)).conjugate(),np.roll(xdir_[dir1_],+1,lOfI[dir2_]))
    tmp = np.einsum('...ij,...jk',tmp,np.roll(np.roll(xdir_[dir2_],+1,lOfI[dir2_]),-1,lOfI[dir1_]))
    tmp = np.einsum('...ij,...jk',tmp,xdir_[dir1_].transpose((0,1,2,3,5,4)).conjugate())
    return tmp


def plaq(xdir_, dir1_="x", dir2_="y"):
    tmp = plaq_general_1(xdir_, dir1_, dir2_)
    tmp = np.sum(tmp, (0,1,2,3))
    return np.einsum('ii', tmp)/(24**3*64*3)


def clover(xdir_, dir1_="x", dir2_="y"):
    ans = plaq_general_1(xdir_, dir1_, dir2_)
    ans = ans + plaq_general_2(xdir_, dir1_, dir2_)
    ans = ans + plaq_general_3(xdir_, dir1_, dir2_)
    ans = ans + plaq_general_4(xdir_, dir1_, dir2_)
    return ans/4


def plaq_clover(xdir_, dir1_="x", dir2_="y"):
    tmp = clover(xdir_, dir1_, dir2_)
    tmp = np.sum(tmp, (0,1,2,3))
    return np.einsum('ii', tmp)/(24**3*64*3)


#FIXME the following functions need to be corrected
'''
def rec_general_1(xdir_, dir1_="x", dir2_="y"):  # horizontal
    tmp = np.einsum('...ij,...jk',xdir_[dir1_],np.roll(xdir_[dir1_],-1,lOfI[dir1_]))
    tmp = np.einsum('...ij,...jk',tmp,np.roll(xdir[dir2_],-2,lOfI[dir1_]))
    tmp = np.einsum('...ij,...jk',tmp,np.roll(np.roll(xdir_[dir1_],-1,lOfI[dir2_]),-1,lOfI[dir1_]).transpose((0,1,2,3,5,4)).conjugate())
    tmp = np.einsum('...ij,...jk',tmp,np.roll(xdir_[dir1_],-1,lOfI[dir2_]).transpose((0,1,2,3,5,4)).conjugate())
    tmp = np.einsum('...ij,...jk',tmp,xdir_[dir2_].transpose((0,1,2,3,5,4)).conjugate())
    return tmp


def rec_general_2(xdir_, dir1_="x", dir2_="y"):  # vertical
    tmp = np.einsum('...ij,...jk',xdir_[dir1_],np.roll(xdir[dir2_],-1,lOfI[dir1_]))
    tmp = np.einsum('...ij,...jk',tmp,np.roll(np.roll(xdir[dir2_],-1,lOfI[dir1_]),-1,lOfI[dir2_]))
    tmp = np.einsum('...ij,...jk',tmp,np.roll(xdir_[dir1_],-2,lOfI[dir2_]).transpose((0,1,2,3,5,4)).conjugate())
    tmp = np.einsum('...ij,...jk',tmp,np.roll(xdir_[dir2_],-1,lOfI[dir2_]).transpose((0,1,2,3,5,4)).conjugate())
    tmp = np.einsum('...ij,...jk',tmp,xdir_[dir2_].transpose((0,1,2,3,5,4)).conjugate())
    return tmp


def rec1(xdir_, dir1_="x", dir2_="y"):
    ans = rec_general_1(xdir_, dir1_, dir2_)
    tmp1 = np.roll(ans, -2, lOfI[dir1_])
    tmp2 = np.roll(ans, -1, lOfI[dir2_])
    ans = ans + tmp1
    ans = ans + tmp2
    ans = ans + np.roll(tmp1, -1, lOfI[dir2_])
    return ans / 4


def rec2(xdir_, dir1_="x", dir2_="y"):
    ans = rec_general_2(xdir_, dir1_, dir2_)
    tmp1 = np.roll(ans, -1, lOfI[dir1_])
    tmp2 = np.roll(ans, -2, lOfI[dir2_])
    ans = ans + tmp1
    ans = ans + tmp2
    ans = ans + np.roll(tmp1, -2, lOfI[dir2_])
    return ans / 4


def rec(xdir_, dir1_="x", dir2_="y"):
    return (rec1(xdir_, dir1_, dir2_) + rec2(xdir_, dir1_, dir2_))/2.


def plaq_rec(xdir_, dir1_="x", dir2_="y"):
    tmp = rec(xdir_, dir1_, dir2_)
    tmp = np.sum(tmp, (0,1,2,3))
    return np.einsum('ii', tmp)/(24**3*64*3)


def symanzik(xdir_, dir1_="x", dir2_="y"):
    return 5./3. * clover(xdir_, dir1_="x", dir2_="y") - 1/12.* rec(xdir_, dir1_="x", dir2_="y")


def iwasaki(xdir_, dir1_="x", dir2_="y"):
    return 3.648 * clover(xdir_, dir1_="x", dir2_="y") - 0.331 * rec(xdir_, dir1_="x", dir2_="y")
'''


# for direct comparison with GWU code control_EB.cpp
def check_EmB(xdir_, x, y, z, t):
    ins = ['x','y','z']
    for i in range(3):
        j=(i+1)%3
        k=(i+2)%3
        tmp1 = clover(xdir_, 't', ins[i])
        tmp1 = tmp1 - tmp1.transpose(0,1,2,3,5,4).conjugate()
        tmp1 /= 2
        tmp1 = tmp1[x,y,z,t,:,:]
        tmp2 = clover(xdir_, ins[j], ins[k])
        tmp2 = tmp2 - tmp2.transpose(0,1,2,3,5,4).conjugate()
        tmp2 /= 2
        tmp2 = tmp2[x,y,z,t,:,:]
        print(i,x,y,z,t,'\n',tmp1@tmp1-tmp2@tmp2)


def E2_plaq(xdir_):
    tmp = plaq_general_1(xdir_, 't', 'x')
    tmp = tmp - tmp.transpose(0,1,2,3,5,4).conjugate()
    tmp /= 2
    ans = np.einsum("...ij,...jk",tmp,tmp)
    
    tmp = plaq_general_1(xdir_, 't', 'y')
    tmp = tmp - tmp.transpose(0,1,2,3,5,4).conjugate()
    tmp /= 2
    ans = ans + np.einsum("...ij,...jk",tmp,tmp)
    
    tmp = plaq_general_1(xdir_, 't', 'z')
    tmp = tmp - tmp.transpose(0,1,2,3,5,4).conjugate()
    tmp /= 2
    ans = ans + np.einsum("...ij,...jk",tmp,tmp)
    
    ans = np.sum(ans, (0,1,2,3))
    return -np.einsum('ii', ans)/(24**3*64)


def B2_plaq(xdir_):
    tmp = plaq_general_1(xdir_, 'x', 'y') 
    tmp = tmp - tmp.transpose(0,1,2,3,5,4).conjugate()
    tmp /= 2
    ans = np.einsum("...ij,...jk",tmp,tmp)
    
    tmp = plaq_general_1(xdir_, 'x', 'z')
    tmp = tmp - tmp.transpose(0,1,2,3,5,4).conjugate()
    tmp /= 2
    ans = ans + np.einsum("...ij,...jk",tmp,tmp)
    
    tmp = plaq_general_1(xdir_, 'y', 'z')
    tmp = tmp - tmp.transpose(0,1,2,3,5,4).conjugate()
    tmp /= 2
    ans = ans + np.einsum("...ij,...jk",tmp,tmp)
    
    ans = np.sum(ans, (0,1,2,3))
    return -np.einsum('ii', ans)/(24**3*64)


def E2_clover_vector(xdir_):
    tmp = clover(xdir_, 't', 'x')
    tmp = tmp - tmp.transpose(0,1,2,3,5,4).conjugate()
    tmp /= 2
    ans = np.einsum("...ij,...ji",tmp,tmp) # trace E2

    tmp = clover(xdir_, 't', 'y')
    tmp = tmp - tmp.transpose(0,1,2,3,5,4).conjugate()
    tmp /= 2
    ans = ans + np.einsum("...ij,...ji",tmp,tmp)
    
    tmp = clover(xdir_, 't', 'z')
    tmp = tmp - tmp.transpose(0,1,2,3,5,4).conjugate()
    tmp /= 2
    ans = ans + np.einsum("...ij,...ji",tmp,tmp)

    return ans.real


def B2_clover_vector(xdir_):
    tmp = clover(xdir_, 'x', 'y')
    tmp = tmp - tmp.transpose(0,1,2,3,5,4).conjugate()
    tmp /= 2
    ans = np.einsum("...ij,...ji",tmp,tmp)  # trace B2

    tmp = clover(xdir_, 'x', 'z')
    tmp = tmp - tmp.transpose(0,1,2,3,5,4).conjugate()
    tmp /= 2
    ans = ans + np.einsum("...ij,...ji",tmp,tmp)
    
    tmp = clover(xdir_, 'y', 'z')
    tmp = tmp - tmp.transpose(0,1,2,3,5,4).conjugate()
    tmp /= 2
    ans = ans + np.einsum("...ij,...ji",tmp,tmp)

    return ans.real


def E2_clover(xdir_):
    ans = E2_clover_vector(xdir_)
    ans = np.sum(ans, (0,1,2,3))
    return ans / -(24**3*64) 


def B2_clover(xdir_):
    ans = B2_clover_vector(xdir_)
    ans = np.sum(ans, (0,1,2,3))
    return ans / -(24**3*64)


check =True

if check:
    st = time.time()
    print('s plaq = ', (plaq(xdir, "x", "y")+plaq(xdir, "x", "z")+plaq(xdir, "y", "z"))/3.)
    print('t plaq = ', (plaq(xdir, "x", "t")+plaq(xdir, "y", "t")+plaq(xdir, "z", "t"))/3.)
    ed = time.time()
    print('used %f s'%(ed - st))

    check_EmB(xdir, 0, 0, 0, 0)
    check_EmB(xdir, 4, 3, 2, 1)
    print("this check_EmB part is carefully checked! GWU uses clover E and B!")

st = time.time()
E2 = E2_clover_vector(xdir)
B2 = B2_clover_vector(xdir)
print(E2.shape)
if check:
    print(E2[0,0,0,0]-B2[0,0,0,0])
    print(E2[4,3,2,1]-B2[4,3,2,1])
ed = time.time()
# for now, the plaq value and the clover E and B are all checked.
print('used %f s'%(ed - st))


st = time.time()
print('E2 plaq = ', E2_plaq(xdir))
ed = time.time()
print('used %f s'%(ed - st))
st = time.time()
print('B2 plaq = ', B2_plaq(xdir))
ed = time.time()
print('used %f s'%(ed - st))
# exit(0)


st = time.time()
print('E2 clover = ', E2_clover(xdir))
ed = time.time()
print('used %f s'%(ed - st))


st = time.time()
print('B2 clover = ', B2_clover(xdir))
ed = time.time()
print('used %f s'%(ed - st))


"""
the following numbers are checked with GWU code (wflow/topo_sus_flow)
Altough I still have no idea why E2_plaq is much larger (around 4 times larger) than E2_clover.
I think I must have missed something.
Again, the numbers are OK. But maybe I need to understand more about E2_plaq.

s plaq =  (0.46778260737941474-8.536204097702349e-05j)
t plaq =  (0.4679082335555097-1.544261084805937e-05j)
used 3.985429 s
E2 plaq =  (5.118906440535736+0j)
used 2.755572 s
B2 plaq =  (5.119680550201284+0j)
used 2.737572 s
E2 clover =  (1.297039024781218-0j)
used 9.444894 s
B2 clover =  (1.2973664570552215-0j)
used 9.582864 s
"""


def E2_clover2(xdir_):
    tmp = clover(xdir_, 't', 'x')
    tmp = 3 - np.einsum('...ii',tmp)
    ans = tmp.real*2

    tmp = clover(xdir_, 't', 'y')
    tmp = 3 - np.einsum('...ii',tmp)
    ans = ans + tmp.real*2
    
    tmp = clover(xdir_, 't', 'z')
    tmp = 3 - np.einsum('...ii',tmp)
    ans = ans + tmp.real*2

    ans = np.sum(ans, (0,1,2,3))
    return ans/(24**3*64)


st = time.time()
print('E2 clover2 = ', E2_clover2(xdir), '\n there should be factor here, 8?')
ed = time.time()
print('used %f s'%(ed - st))


"""
this part is quiet interesting.
I actually have more things to check,
like whether the FmunuFmunu part is a constant times unitray matrix...
"""


# format of noise vector of GWU
# r/i t z y x, already checked

E2 = E2_clover_vector(xdir) 
print(E2.shape)
E2 = np.concatenate([E2.real.reshape(1,64,24,24,24), E2.imag.reshape(1,64,24,24,24)])
print(E2.shape)
E2.tofile('E2_test.bin')
E2_new = np.fromfile('E2_test.bin')
E2_new = E2_new.reshape(E2.shape)
print(np.sum(np.abs(E2_new-E2)))

B2 = B2_clover_vector(xdir) 
print(B2.shape)
B2 = np.concatenate([B2.real.reshape(1,64,24,24,24), B2.imag.reshape(1,64,24,24,24)])
print(B2.shape)
B2.tofile('B2_test.bin')
B2_new = np.fromfile('B2_test.bin')
B2_new = B2_new.reshape(B2.shape)
print(np.sum(np.abs(B2_new-B2)))

exit(0)

