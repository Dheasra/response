from vampyr import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt
# from copy import deepcopy

from scfsolv import scfsolv as ss

class KAIN:
    def __init__(self, history) -> None:
        self.history = history

#TODO: faire le expandsolution

    def expandSolution(phi_prev, f_prev):
        #Compute the fock matrix of the system
        F, V, J, K = compFock(phi_prev) #TODO: changer le reste pour que ça corresponde avec ec nouveau formalisme
        
        phistory = []
        E_n = []
        norm = []
        update = []
        for orb in range(nOrb):
            E_n.append(F[orb, orb])
            #Redefine the Helmholtz operator with the updated energy
            mu = np.sqrt(-2*E_n[orb])
            G_mu = vp.HelmholtzOperator(mra, mu, prec) 

            #Compute phi_tmp := Sum_{j!=i} F_ij*phi_j 
            phi_tmp = P_eps(Fzero)
            for orb2 in range(nOrb):
                #Compute off-diagonal Fock matrix elements
                if orb2 != orb:
                    phi_tmp = phi_tmp + F[orb, orb2]*phi_prev[orb2][-1]
            #Compute new power iteration for the Helmholtz operatort 
            phi_np1 = -2*G_mu((V + J)*phi_prev[orb][-1] - K[orb] - phi_tmp)
            #create an alternate history of orbitals which include the power iteration
            phistory.append([phi_np1])
        #Orthonormalise the alternate orbital history
        phistory = orthonormalise(phistory)
        # phi_prev = orthonormalise(phi_prev)
        for orb in range(nOrb):
            f_prev[orb].append(phistory[orb][-1] - phi_prev[orb][-1])
            #Setup and solve the linear system Ac=b
            c = setuplinearsystem(phi_prev[orb], f_prev[orb])
            #Compute the correction delta to the orbitals 
            delta = f_prev[orb][-1]
            for j in range(len(phi_prev[orb])-1):
                delta = delta + c[j]*(phi_prev[orb][j] - phi_prev[orb][-1] + f_prev[orb][j] - f_prev[orb][-1])
            #Apply correction
            phi_n = phi_prev[orb][-1]
            phi_n = phi_n + delta
            #Normalize
            norm.append(phi_n.norm())
            phi_n.normalize()
            #Save new orbital
            phi_prev[orb].append(phi_n)
            #Correction norm (convergence metric)
            update.append(delta.norm())
        return phi_prev, f_prev, np.array(E_n), np.array(norm), np.array(update)
    
    def setuplinearsystem(phi_prev, f_prev):
        lenHistory = len(phi_prev)
        # Compute matrix A
        A = np.zeros((lenHistory-1, lenHistory-1))
        b = np.zeros(lenHistory-1)
        for l in range(lenHistory-1):  
            dPhi = phi_prev[l] - phi_prev[-1]
            b[l] = vp.dot(dPhi, f_prev[-1])
            for j in range(lenHistory-1):
                A[l,j] = -vp.dot(dPhi, f_prev[j] - f_prev[-1])
        #solve Ac = b for c
        c = np.linalg.solve(A, b)
        return c