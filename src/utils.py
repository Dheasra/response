from vampyr import vampyr3d as vp
import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from scipy.special import eval_genlaguerre

from spinor import spinor

#Stuff to import complex_fcn from far away
import sys
import os
# Construct the absolute path to the directory containing the module.
module_path = os.path.abspath("/home/qpitto/Tests_KAIN/ZORA/ReMRChem2C/orbital4c")
# Append the module path to sys.path
sys.path.append(module_path)
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
        output.compVect[i] = apply_Poisson_complex(pois_op, spin.compVect[i])
    return output

def  apply_Pauli(direction, spinr):
    output = spinor(spinr.mra, len(spinr))
    # print("Pauli start")
    if len(spinr) == 2:
        if direction == 0:
            # sigma = np.array([0, 1], [1, 0])
            # tmpTree = spinr.compVect[0]
            output.compVect[0] = spinr.compVect[1]
            output.compVect[1] = spinr.compVect[0]
            # print("Pauli x")
        elif direction == 1:
            # sigma = np.array([0, -1j], [1j, 0])
            # tmpTree = spinr.compVect[0]
            output.compVect[0] = -1j*spinr.compVect[1]
            output.compVect[1] = 1j*spinr.compVect[0]
            # print("Pauli y")
        else:
            # sigma = np.array([1, 0], [0, -1])
            output.compVect[0] = spinr.compVect[0]
            output.compVect[1] = -1*spinr.compVect[1]
            # print("Pauli z")
    else:
        raise ValueError("1- or 4-component spinor not supported by multiplication of Pauli matrices")
    # print("Pauli done")
    return output
    

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

def is_constant(cplx_fct, threshold = 1e-8, target_value = None): 
    r_x = np.linspace(-5., 5., 1000)
    realx = [cplx_fct.real([x, 0.0, 0.0]) for x in r_x]
    avrgx = np.average(realx)
    realy = [cplx_fct.real([0.0, x, 0.0]) for x in r_x]
    avrgy = np.average(realy)
    realz = [cplx_fct.real([0.0, 0.0, x]) for x in r_x]
    avrgz = np.average(realz)
    imagx = [cplx_fct.imag([x, 0.0, 0.0]) for x in r_x]
    avrgix = np.average(imagx)
    imagy = [cplx_fct.imag([0.0, x, 0.0]) for x in r_x]
    avrgiy = np.average(imagy)
    imagz = [cplx_fct.imag([0.0, 0.0, x]) for x in r_x]
    avrgiz = np.average(imagz)
    if target_value != None:
        avrgx, avrgy, avrgz, avrgix, avrgiy, avrgiz = target_value, target_value, target_value, target_value, target_value, target_value
    for i in range(1,len(r_x)):
        if np.abs(realx[i]-avrgx) > threshold:
            return False
        if np.abs(realy[i]-avrgy) > threshold:
            return False
        if np.abs(realz[i]-avrgz) > threshold:
            return False
        if np.abs(imagx[i]-avrgix) > threshold:
            return False
        if np.abs(imagy[i]-avrgiy) > threshold:
            return False
        if np.abs(imagz[i]-avrgiz) > threshold:
            return False
    return True



# Shamelessly stolen with permission from https://github.com/ilfreddy/ReMRChem/blob/NR-starting-guess/starting_guess.py
def make_NR_starting_guess_Hlike(position, charge, mra, prec, Ncompo = 2):
    nr_wf_tree = vp.FunctionTree(mra)
    nr_wf_tree.setZero()
    n = 1
    l = 0
    Peps = vp.ScalingProjector(mra, prec)
    guess = lambda x : wf_hydrogenionic_atom(n,l,[x[0]-position[0], x[1]-position[1], x[2]-position[2]],charge)
    nr_wf_tree = Peps(guess)
    
    La_comp = complex_fcn(mra)
    La_comp.real = nr_wf_tree

    spinorb1 = spinor(mra, N_components=Ncompo)
    # spinorb2 = spinor(mra, N_components=2)
    spinorb1.compVect[0] = La_comp
    # spinorb1.init_small_components(prec/10)
    spinorb1.normalize()
    spinorb1.crop(prec)
    return spinorb1

#returns the value of the radial WF in the point r
# 1. the nucleus is assumed infintely heavy (mass of electron and Bohr radius used)
# 2. the nucleus is placed in the origin
# 3. atomic units are assumed a0 = 1  hbar = 1  me = 1  4pie0 = 1
def radial_wf_hydrogenionic_atom(n,l,r,Z):
    rho = 2 * Z * r
    slater = np.exp(-rho/2)
    polynomial = eval_genlaguerre(n-l-1, 2*l+1, rho)
    f1 = np.math.factorial(n-l-1)
    f2 = np.math.factorial(n+l)
    norm = np.sqrt((2*Z/n)**3 * f1 / (2 * n * f2))
    value = norm * rho**l * polynomial * slater
    return value

def wf_hydrogenionic_atom(n,l,position,Z):
    if(l != 0):
        print("only s orbitals for now")
        exit(-1)
    distance = np.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
    value = radial_wf_hydrogenionic_atom(n, l, distance, Z)
    return value

def make_NR_starting_guess(positionList, chargeList, mra, prec, Ncompo=2):
    initGuess = spinor(mra, Ncompo)
    initGuess.setZero()
    for r in range(len(positionList)):
        initGuess = initGuess + make_NR_starting_guess_Hlike(positionList[r], chargeList[r], mra, prec)
    initGuess.normalize()
    initGuess.crop(prec)
    return initGuess