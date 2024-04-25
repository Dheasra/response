from vampyr import vampyr3d as vp
import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from spinor import spinor
from complex_fcn import complex_fcn


# zero function
def Fzero(r):
    return 0.

def Fone(r):
    return 1.

# #Computes an integral of the type \int_{\mathbb{R}^3}\text{d}^3r f(r)\vec{r} using a trapezoidal approximation of f 
# def directionIntegration(fTree, world, rDir, nSample = 1000):
#     #fTree is the integrand function f in Function Tree representation
#     #rDir is the direction of the vector r (not necessarily normalised)
#     #world contains information of the computational world, including the boundaries which are needed here

#     upBnds = world.upperBounds
#     lwBnds = world.lowerBounds
#     for direction in len(rDir):
#         if rDir[direction] != 0:
#             print("todo")
#             r_i = np.linspace(-5., 5., 1000)
#             # for 
#             #TODO: faire un algo d'intégration


#En fait je pourrais quand même utiliser une f_tree avec flin 
def Flin(r, Direction):
        return r[Direction]

def Fx(r):
     return r[0]

def apply_Poisson_complex(pois_op, c_fct):
     output = complex_fcn(c_fct.mra)
     output.real = pois_op(c_fct.real)
     output.imag = pois_op(c_fct.imag)
     return output

def apply_Poisson_spinor(pois_op, spin): 
    output = spinor(spin.mra,len(spin))
    for i in range(len(spin)):
        output.orbVect[i] = apply_Poisson_complex(pois_op, spin.orbVect[i])
    return output

def  apply_Pauli(direction, spinr):
    if len(spinr) == 2:
        if direction == 0:
            # sigma = np.array([0, 1], [1, 0])
            tmpTree = spinr.compVect[0]
            spinr.compVect[0] = spinr.compVect[1]
            spinr.compVect[1] = tmpTree
        elif direction == 1:
            # sigma = np.array([0, -1j], [1j, 0])
            tmpTree = spinr.compVect[0]
            spinr.compVect[0] = -1j*spinr.compVect[1]
            spinr.compVect[1] = 1j*tmpTree
        else:
            # sigma = np.array([1, 0], [0, -1])
            spinr.compVect[1] = -1*tmpTree
    else:
        raise ValueError("4-component spinor not supported in this implementation")
    

def LeviCivita(indices): #Returns the value of the Levi-Civita tensor. 
    #Indicies is an array containing the indices of the tensor we want to sample. For instance, [1 2 3] would return 1 because espilon^(1 2 3) is 1 and [2 1 3] would be -1
    output = 1
    # print("pouet")
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            # print(-indices[i] + indices[j])
            output *= (-indices[i] + indices[j])
    output *= 1/math.factorial(len(indices)-1) 
    if len(indices)>1:
        output *= 1/math.factorial(len(indices)-2)
    return output