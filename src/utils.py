from vampyr import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def f_nuc(r):
    out = 0
    #electron-nucleus interaction
    for i in range(nz): 
        out += -Z[i]/np.sqrt((r[0]-R[i][0])**2 + (r[1]-R[i][1])**2 + (r[2]-R[i][2])**2) 
    return out

def f_NN():
    #nucleus-nucleus interaction
    out = 0
    # print("zblim")
    for i in range(nz-1):
        # print("zboum")
        for j in range(i+1, nz):
            # print(i, j)
            out += Z[i]*Z[j]/np.sqrt((R[j][0]-R[i][0])**2 + (R[j][1]-R[i][1])**2 + (R[j][2]-R[i][2])**2)
    return out 

# zero function
def Fzero(r):
    return 0.