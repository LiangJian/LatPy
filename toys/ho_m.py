import numpy as np
import sys
import gvar as gv
import numba

Nt = 100
m = 1
w = 1

dt = 0.1
x = np.zeros(Nt) 

# action
def s(m_, x_, dt_, w_):
    return np.sum( 0.5 * m_ * (np.roll(x_, -1) - x_) ** 2 / dt_ / dt_ + 0.5 * m * w_ ** 2 * x_ ** 2 ) * dt_


Nc = 20000  # number of trajectories
Xs = []  # configurations
dd = 0.2  # step
dc = 100  # use every dc trajectories

S_old = np.inf
i = 0

while i < Nc:
    x_ = x
    x_ = x_ + ( 2 * np.random.rand(Nt) - 1 ) * dd
    x_[0] = x_[Nt-1]
    S_new = s(m, x_, dt, w)
    if S_new <= S_old:
        S_old = S_new
        x = x_
        i += 1
        Xs.append(x)
        if i%dc == 0:
            print(i,end='\t')
            sys.stdout.flush()
    elif S_new > S_old:
        p = np.random.rand()
        if np.exp(S_old - S_new) > p:
            S_old = S_new
            x = x_
            i += 1
            Xs.append(x)
            if i%dc == 0:
                print(i,end='\t')
                sys.stdout.flush()

print()
Xs = np.array(Xs).reshape(Nc//dc, dc, Nt)
cut = 10  # thermalization
Xs = Xs[cut:,...]
Xs = Xs[:, 0, :]
X2 = Xs ** 2
print(gv.gvar(np.average(X2,0),np.std(X2,0)/np.sqrt(X2.shape[0])), end='\t')
print(gv.gvar(np.average(X2),np.std(X2)/np.sqrt(X2.size)), end='\t')

C2 = [np.sum(Xs * np.roll(Xs,-_,1), 1) for _ in range(15)]
C2 = np.array(C2)
print(C2.shape)
C2_sum = np.sum(C2, 1)
# print(C2_sum)
# print(C2_sum/np.roll(C2_sum, -1))

C2 = (C2_sum.reshape(C2.shape[0], 1) - C2) / (C2.shape[1] - 1)
E2 = np.log(C2 / np.roll(C2, -1, 0)) / dt
print(gv.gvar(np.average(E2, 1),np.std(E2, 1)*np.sqrt(E2.shape[1])), end='\t')

