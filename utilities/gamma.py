import numpy as np

g1 = np.matrix([[   0,   0,   0, -1j],
                [   0,   0, -1j,   0], 
                [   0, +1j,   0,   0], 
                [ +1j,   0,   0,   0]])

g2 = np.matrix([[   0,   0,   0,  -1],
                [   0,   0,  +1,   0], 
                [   0,  +1,   0,   0], 
                [  -1,   0,   0,   0]])

g3 = np.matrix([[   0,   0, -1j,   0],
                [   0,   0,   0, +1j], 
                [ +1j,   0,   0,   0], 
                [   0, -1j,   0,   0]])

g4 = np.matrix([[  +1,   0,   0,   0],
                [   0,  +1,   0,   0], 
                [   0,   0,  -1,   0], 
                [   0,   0,   0,  -1]])

g5 = np.matrix([[   0,   0,  +1,   0],
                [   0,   0,   0,  +1], 
                [  +1,   0,   0,   0], 
                [   0,  +1,   0,   0]])

g0 = np.matrix([[  +1,   0,   0,   0],
                [   0,  +1,   0,   0], 
                [   0,   0,  +1,   0], 
                [   0,   0,   0,  +1]])

gi = [g0, g1, g2, g3, g4, g5]
gip = []
giv = []
for ig in gi:
    for i in range(4):
        for j in range(4):
            if ig[i, j] != 0.0:
                gip.append(j)
                giv.append(ig[i, j])

gip = np.array(gip)
giv = np.array(giv)
gip = gip.reshape(6, 4)
giv = giv.reshape(6, 4)

g15 = g1*g5
g25 = g2*g5
g35 = g3*g5
g45 = g4*g5

g51 = -g15
g52 = -g25
g53 = -g35
g54 = -g45

g5i = [g51, g52, g53, g54]
g5ip = []
g5iv = []
for ig in g5i:
    for i in range(4):
        for j in range(4):
            if ig[i, j] != 0.0:
                g5ip.append(j)
                g5iv.append(ig[i, j])

g5ip = np.array(g5ip)
g5iv = np.array(g5iv)
g5ip = g5ip.reshape(4, 4)
g5iv = g5iv.reshape(4, 4)

gi5 = [g15, g25, g35, g45]
gi5p = []
gi5v = []
for ig in gi5:
    for i in range(4):
        for j in range(4):
            if ig[i, j] != 0.0:
                gi5p.append(j)
                gi5v.append(ig[i, j])

gi5p = np.array(gi5p)
gi5v = np.array(gi5v)
gi5p = gi5p.reshape(4, 4)
gi5v = gi5v.reshape(4, 4)

g = gi + gi5
gp = np.concatenate([gip,gi5p])
gv = np.concatenate([giv,gi5v])

