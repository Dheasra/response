from vampyr import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


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