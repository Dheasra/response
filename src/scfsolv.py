from vampyr import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt
# from copy import deepcopy


#Stuff to import complex_fcn from far away
import sys
import os
# Construct the absolute path to the directory containing the module.
module_path = os.path.abspath("/home/qpitto/Tests_KAIN/ZORA/ReMRChem2C/orbital4c")
# Append the module path to sys.path
sys.path.append(module_path)

# import KAIN
import utils
from complex_fcn import complex_fcn, apply_poisson
from spinor import spinor

#Class containing all the methods necessary to compute and optimise ground-state orbitals
class scfsolv:
    world : vp.BoundingBox
    mra : vp.MultiResolutionAnalysis
    prec : float
    P_eps : vp.ScalingProjector             #projection operator
    Pois : vp.PoissonOperator               #Poisson operator
    D : vp.ABGVDerivative                   #derivative operator
    G_mu : list                             #list of Helmholtz operator for every orbital
    Norb : int                              #number of orbitals
    Fock : np.ndarray                       #Fock matrix
    Vnuc : vp.FunctionTree                  #nuclear potential
    J : vp.FunctionTree                     #Coulomb potential
    K : list                                #list of exchange potential applied to each orbital
    phi_prev : list                         #list of all orbitals and their KAIN history
    f_prev : list                           #list of all orbitals updates and their KAIN history 
    khist : int                             #KAIN history size 
    R : list                                #list of all coordinates of each atom
    Z : list                                #list of all atomic numbers of each atom
    Nz : int                                #number of atoms
    E_pp : float                            #internal nuclear energy
    E_n : list                              #List of orbital energies
    c : float                               #Light speed in atomic units
    Ncomp : int                             #Number of components of the spinors


    # Constructor of the class
    # Initialise the class with placeholder default values for the physical variables, regardless of the physical system.
    # The MRA (polynomial order, grid size, etc), operators (Poisson, derivative, projection) are customised in this method, and can not manually be changed afterwards.
    # The KAIN accelerator history is also set in this method, and can neither be changed manually.
    # prec[in]: floating point number setting the desired precision of the operators, notably it sets the precision of projected functions onto the MRA.
    # khist[in]: integer setting the history length for the KAIN solver. 
    # lgdrOrder[in]: integer setting the Legendre polynomial order used to represent functions in the MRA.
    # sizeScale[in]: integer setting the unit scale of each box
    # nboxes[in]: integer setting the initial number of boxes. Remember that in 3d, space must be discretised in boxes.
    # scling[in]: floating point number doing something that I don't remember what it is. I confuse it with scale. Leaving it at 1.0 never failed me though.
    def __init__(self, Ncomponents, prec, khist, lgdrOrder=6, sizeScale=-4, nboxes=2, scling=1.0) -> None:
        self.world = vp.BoundingBox(corner=[-1]*3, nboxes=[nboxes]*3, scaling=[scling]*3, scale= sizeScale)
        self.mra = vp.MultiResolutionAnalysis(order=lgdrOrder, box=self.world)
        self.prec = prec
        self.P_eps = vp.ScalingProjector(self.mra, self.prec) 
        self.Pois = vp.PoissonOperator(self.mra, self.prec)
        #Derivative operator
        self.D = vp.ABGVDerivative(self.mra, a=0.5, b=0.5)
        #Physical properties (placeholder values)
        self.Norb = 1
        self.Fock = np.zeros((self.Norb, self.Norb), dtype=complex)
        # initialise complex-valued function trees
        self.J = spinor(self.mra, Ncomponents)
        self.J.setZero()
        # # Vnuc only has a real part, but I am not creating a real-valued spinor class
        self.Vnuc = spinor(self.mra, Ncomponents)
        self.Vnuc.setZero()
        # self.Vnuc = self.P_eps(utils.Fzero)
        self.K = []
        self.R = []
        self.Z = []
        self.Nz = 1
        self.E_pp = 0.0
        #Accelerator
        self.khist = khist
        self.phi_prev = []
        self.f_prev = []
        # #ZORA
        self.c = 137.02 #in atomic units
        self.Ncomp = Ncomponents
        # testZORA = complex_fcn(self.mra)

    # This method initialises all physical properties of the system.
    # No[in]: integer giving the number of orbitals. Be warned that this code currently only works for closed shell systems.
    # pos[in]: List of lists (Matrix-ish) of floating point number. The mother list contains lists (vectors) of the 3D coordinates of the each atom in the system, in atomic units.
    # Z[in]: List of integers. Each number is the atomic charge of an atom in the system. Be warned that each atoms must be ordered in the same way in pos and Z
    # init_g_dir[in]: Directory (string) of the inital guesses for the orbitals. Currently this code cannot create an initial guess, and thus needs an MRChem-compatible inital guess created from another program, such as MRChem.
    def initMolec(self, No,  pos, Z, init_g_dir) -> None: 
        self.Nz = len(Z)
        self.Norb = No
        self.R = pos
        self.Z = Z
        for i in range(len(self.Vnuc)):
            self.Vnuc.compVect[i].real = self.P_eps(lambda r : self.f_nuc(r))
        #initial guesses provided by mrchem
        self.phi_prev = []
        for i in range(self.Norb):
            phi = spinor(self.mra, self.Ncomp) #TODO: CONTINUER À PARTIR D'ICI
            phi.setZero()
            phi.compVect[0].real.loadTree(f"{init_g_dir}phi_p_scf_idx_{i}_re")  #TODO 2C
            self.phi_prev.append([phi])
        self.f_prev = [[] for i in range(self.Norb)] #list of the corrections at previous steps
        # print("Overlap init")
        # print("S=",self.computeOverlap())
        # TestImag = 1j*self.phi_prev[0][-1]
        # print("InitMolec test imag: ", complex_fcn.dot(self.phi_prev[0][-1],TestImag))
        #Compute the Fock matrix and potential operators 
        self.compFock()
        # print("Fock = ", self.Fock)
        #Energies of each orbital
        self.E_n = []
        #internal nuclear energy
        self.E_pp = self.fpp()
        # Prepare Helmholtz operators
        self.G_mu = []
        # print("Overlap init post-fock")
        # print("S=",self.computeOverlap())
        for i in range(self.Norb):
            self.E_n.append(self.Fock[i,i])
            # print("Energy: ", self.Fock[i, i], i)
            mu = np.sqrt(-2*self.E_n[i])
            # print("mu", mu)
            self.G_mu.append(vp.HelmholtzOperator(self.mra, mu, self.prec))  # Initalize the operator
        for orb in range(self.Norb):
            #First establish a history (at least one step) of corrections to the orbital with standard iterations with Helmholtz operator to create
            # Apply Helmholtz operator to obtain phi_np1 #5
            phi_np1 = self.powerIter(orb)
            # Compute update = ||phi^{n+1} - phi^{n}|| #6
            # print("Post power Iter, pre normalize", orb, complex_fcn.dot(phi_np1, phi_np1))
            self.f_prev[orb].append(phi_np1 - self.phi_prev[orb][-1])
            phi_np1.normalize()
            # print("Post power Iter, post normalize", orb, complex_fcn.dot(phi_np1, phi_np1))
            self.phi_prev[orb].append(phi_np1)
        # print("Overlap init post-powerIter")
        # print("S=",self.computeOverlap())
        #Orthonormalise orbitals since they are molecular orbitals
        phi_ortho = self.orthonormalise()
        for orb in range(self.Norb): #Mandatory loop due to questionable data format choice.
            self.phi_prev[orb][-1] = phi_ortho[orb]
        # print("Overlap init end")
        # print("S=",self.computeOverlap())

    #===Computation of operators===
    
    # This method computes the Fock matrix of the system
    def compFock(self): 
        self.Fock = np.zeros((self.Norb, self.Norb), dtype=complex)
        self.K = []
        self.J = self.computeCoulombOperator()
        for j in range(self.Norb):
            #Compute the potential operator
            self.K.append(self.computeExchangePotential(j))            
            Fphi = self.compFop(j)
            # compute the energy from the orbitals 
            for i in range(self.Norb):
                # self.Fock[i,j] = complex_fcn.dot(self.phi_prev[i][-1], Fphi) #TODO 2C
                self.Fock[i,j] = self.phi_prev[i][-1].dot(Fphi) 

    # This method computes the Fock operator applied to an orbital "orb"; returns F_\phi = \hat{F}\ket{\phi} 
    # orb[in]: integer index of the chosen orbital
    # Fphi[out]: function tree (vp.FunctionTree) representation of the operator applied to \ket{\phi[idx]}
    def compFop(self, orb): 
        #Zora potential
        V_z = self.Vnuc + self.J
        #constant fct f(x) = 1
        one = spinor(self.mra, self.Ncomp)
        one.setZero()
        for j in range(self.Ncomp):
            one.compVect[j].real = self.P_eps(utils.Fone)
        #kappa operator
        kappa = (one-V_z/(2*self.c**2))**(-1) #TODO: There will be an error here because Integer - ftree is probably not defined
        #Kinetic operator computation 
        #first scalar term: -0.5*kappa*nabla^2
        kNab2 = -0.5 * kappa * self.phi_prev[orb][-1].derivative(0).derivative(0)
        kNab2 = kNab2 - 0.5 * kappa * self.phi_prev[orb][-1].derivative(1).derivative(1)
        kNab2 = kNab2 -  0.5 * kappa * self.phi_prev[orb][-1].derivative(2).derivative(2)
        #second scalar term: -0.5 *nabla(kappa) * nabla
        NabkNab = -0.5 * kappa.derivative(0) * self.phi_prev[orb][-1].derivative(0)
        NabkNab = NabkNab -0.5 * kappa.derivative(1) * self.phi_prev[orb][-1].derivative(1)
        NabkNab = NabkNab -0.5 * kappa.derivative(2) * self.phi_prev[orb][-1].derivative(2)
        #--spin orbit term--
        #x direction
        sporb = 1j * utils.apply_Pauli(0,-0.5*(kappa.derivative(1)*self.phi_prev[orb][-1].derivative(2)-kappa.derivative(2)*self.phi_prev[orb][-1].derivative(1)))
        sporb = sporb + 1j * utils.apply_Pauli(1,-0.5*(kappa.derivative(2)*self.phi_prev[orb][-1].derivative(0)-kappa.derivative(0)*self.phi_prev[orb][-1].derivative(2)))
        sporb = sporb + 1j * utils.apply_Pauli(2,-0.5*(kappa.derivative(0)*self.phi_prev[orb][-1].derivative(1)-kappa.derivative(1)*self.phi_prev[orb][-1].derivative(0)))
        #Total kinetic operator
        Tphi = kNab2 + NabkNab + sporb
        # print("types CompFop", type(self.Vnuc), type(self.J), type(self.phi_prev[orb][-1]), self.K[orb])
        Fphi = Tphi + self.Vnuc * self.phi_prev[orb][-1] + self.J * self.phi_prev[orb][-1] - self.K[orb]
        # print("CompFop", orb, complex_fcn.dot(self.J * self.phi_prev[orb][-1], self.J * self.phi_prev[orb][-1]))
        # print("CompFop real", orb, vp.dot(Fphi.real, Fphi.real))
        return Fphi

    #Computes the product between an electron in orbital orb1 and another in orbital orb2
    # orb1[in]: integer index of the left (bra) element of the product 
    # orb2[in]: integer index of the right (ket) element of the product
    # rho[out]: function tree (vp.FunctionTree) representation of the product
    def compProduct(self, orb1, orb2): 
        #NOTE: to compute the density, a factor 2 will be lacking, as we are counting over full orbitals, not electrons
        # but in this code it is added where needed
        rho = self.phi_prev[orb1][-1]*self.phi_prev[orb2][-1]
        return rho

    #This method computes the Coulomb operator  of the molecule
    # [out]: function tree (vp.FunctionTree) representation of the operator
    def computeCoulombOperator(self): 
        # output = spinor(self.mra, self.Ncomp)
        PNbr = 4*np.pi*self.compProduct(0, 0)
        for orb in range(1, self.Norb):
            PNbr = PNbr + 4*np.pi*self.compProduct(orb, orb)
        # return self.Pois(2*PNbr) #factor of 2 because we sum over the number of orbitals, not electrons
        # output.real = self.Pois(2*PNbr.real)
        # output.imag = self.Pois(2*PNbr.imag)
        return utils.apply_Poisson_spinor(self.Pois, PNbr)
        # return output
    

    #This method computes the exchange operator applied to an orbital of index "idx"
    # idx[in]: integer index of the chosen orbital in the list of orbitals "phi_prev"
    # [out]: function tree (vp.FunctionTree) representation of the operator
    def computeExchangePotential(self, idx):
        # K = complex_fcn(self.mra)
        # Pois_term = complex_fcn(self.mra)
        # Pois_term.real = self.Pois(4*np.pi*self.compProduct(0, idx).real)
        # Pois_term.imag = self.Pois(4*np.pi*self.compProduct(0, idx).imag)
        # K.real = (self.phi_prev[0][-1]*Pois_term).real
        # K.imag = (self.phi_prev[0][-1]*Pois_term).imag
        # for j in range(1, self.Norb):
        #     Pois_term.real = self.Pois(4*np.pi*self.compProduct(j, idx).real)
        #     Pois_term.imag = self.Pois(4*np.pi*self.compProduct(j, idx).imag)
        #     K.real = K.real + (self.phi_prev[j][-1]*Pois_term).real
        #     K.imag = K.imag + (self.phi_prev[j][-1]*Pois_term).imag
        #     # K = K + self.phi_prev[j][-1]*self.Pois(4*np.pi*self.compProduct(j, idx))
        # return K 
        K = spinor(self.mra, self.Ncomp)
        K.setZero()
        for j in range(self.Norb):
            K = self.phi_prev[j][-1] * utils.apply_Poisson_spinor(self.Pois, 4*np.pi*self.compProduct(j, idx))
        return K
        # K = self.phi_prev[0][-1] * complex_fcn.apply_poisson(4 * np.pi * self.phi_prev[0][-1].density(self.prec), self.mra, self.Pois, self.prec)
        # for j in range(1, self.Norb):
        #     K += self.phi_prev[j][-1] * complex_fcn.apply_poisson(4 * np.pi * self.phi_prev[j][-1].density(self.prec), self.mra, self.Pois, self.prec)
        # return K
    
    #TODO: continuer ici

    #This is the main method of the class; it moves a step forward in the optimisation of the orbitals
    #It makes use of a KAIN accelerator to improves convergence speed.
    #The updated orbitals and history are stored in the class' attributes, and deleted when too old in the history.
    #E_n[out]: Energy level of each orbital in atomic units, outputed in a numpy array
    #norm[out]: norm of each orbital, given in a numpy array
    #update[out]: norm of the difference between the newest and second newest orbital in the history, given in a numpy array.
    def expandSolution(self):
        #Orthonormalise orbitals in case they aren't yet
        phi_ortho = self.orthonormalise()
        for orb in range(self.Norb): #Mandatory loop due to questionable data format choice.
            self.phi_prev[orb][-1] = phi_ortho[orb]
        #Compute the fock matrix of the system
        self.compFock()
        
        phistory = []
        self.E_n = []
        norm = []
        update = []
        for orb in range(self.Norb):
            self.E_n.append(self.Fock[orb, orb])
            #Redefine the Helmholtz operator with the updated energy
            mu = np.sqrt(-2*self.E_n[orb])
            self.G_mu[orb] = vp.HelmholtzOperator(self.mra, mu, self.prec) 
            #Compute new power iteration for the Helmholtz operator
            phi_np1 = self.powerIter(orb)        
            #create an alternate history of orbitals which include the power iteration
            phistory.append([phi_np1])
        #Orthonormalise the alternate orbital history
        phistory = self.orthonormalise(phistory)
        # phi_prev = orthonormalise(phi_prev)
        for orb in range(self.Norb):
            self.f_prev[orb].append(phistory[orb] - self.phi_prev[orb][-1])
            #Setup and solve the linear system Ac=b
            c = self.setupLinearSystem(orb)
            #Compute the correction delta to the orbitals 
            delta = self.f_prev[orb][-1]
            for j in range(len(self.phi_prev[orb])-1):
                delta = delta + c[j]*(self.phi_prev[orb][j] - self.phi_prev[orb][-1] + self.f_prev[orb][j] - self.f_prev[orb][-1])
            #Apply correction
            phi_n = self.phi_prev[orb][-1]
            phi_n = phi_n + delta
            #Normalize
            # norm_Re  = phi_n.real.norm()
            # norm_Im  = phi_n.imag.norm()
            # norm.append(np.sqrt(norm_Re**2 + norm_Im**2))
            norm.append(np.sqrt(phi_n.squaredNorm()))
            phi_n.normalize()
            #Save new orbital
            self.phi_prev[orb].append(phi_n)
            #Correction norm (convergence metric)
            update.append(np.sqrt(delta.squaredNorm())) 
            if len(self.phi_prev[orb]) > self.khist: #deleting oldest element to save memory
                del self.phi_prev[orb][0]
                del self.f_prev[orb][0]
        return np.array(self.E_n), np.array(norm), np.array(update)
    
    #This method executes one SCF "power iteration", i.e. one application of the Helmholtz operator to the SCF equation
    #orb[in]: integer index of the chosen orbital in the list of orbitals "phi_prev"
    #[out]: function tree (vp.FunctionTree) representation of the updated orbital with index "orb"
    def powerIter(self, orb):
         #Zora potential
        V_z = self.Vnuc + self.J
        #constant fct f(x) = 1
        one = spinor(self.mra, self.Ncomp)
        one.setZero()
        for j in range(self.Ncomp):
            one.compVect[j].real = self.P_eps(utils.Fone)
        #kappa operator
        kappa_m1 = one-V_z/(2*self.c**2)
        kappa = kappa_m1**(-1)

        #First SCF term (Scalar kinetic term)
        Term1 = spinor(self.mra, self.Ncomp)
        Term1.setZero()
        for i in range(3): 
            Term1 = Term1 -1*kappa_m1*kappa.derivative(i)*self.phi_prev[orb][-1].derivative(i)
        
        #Second SCF term (spin-orbit kinetic term)
        # T2 = spinor(self.mra, self.Ncomp)
        # T2.setZero()
        Term2 = -1j * kappa_m1 * utils.apply_Pauli(0,-0.5*(kappa.derivative(1)*self.phi_prev[orb][-1].derivative(2)-kappa.derivative(2)*self.phi_prev[orb][-1].derivative(1)))
        Term2 = Term2 - 1j * kappa_m1 * utils.apply_Pauli(1,-0.5*(kappa.derivative(2)*self.phi_prev[orb][-1].derivative(0)-kappa.derivative(0)*self.phi_prev[orb][-1].derivative(2)))
        Term2 = Term2 - 1j * kappa_m1 * utils.apply_Pauli(2,-0.5*(kappa.derivative(0)*self.phi_prev[orb][-1].derivative(1)-kappa.derivative(1)*self.phi_prev[orb][-1].derivative(0)))
        
        #Third SCF term (potential)
        Vkphi = (self.Vnuc + self.J)*kappa_m1*self.phi_prev[orb][-1] - self.K[orb]*kappa_m1
        VzFphi = V_z/(2*self.c**2) * self.Fock[orb, orb] * self.phi_prev[orb][-1] 
        Term3 = Vkphi + VzFphi
        
        #Fourth SCF term (Non-canonical basis correction)
        Term4 = self.P_eps(utils.Fzero) 
        for orb2 in range(self.Norb):
            #Compute off-diagonal Fock matrix elements
            if orb2 != orb:
                Term4 = Term4 + self.Fock[orb, orb2]*self.phi_prev[orb2][-1]
        Term4 = Term4*kappa_m1


        
        # #TODO: implémenter l'équation de SCF
        # # #Compute phi_ortho := Sum_{j!=i} F_ij*phi_j 
        # # phi_ortho = self.P_eps(utils.Fzero) 
        # # for orb2 in range(self.Norb):
        # #     #Compute off-diagonal Fock matrix elements
        # #     if orb2 != orb:
        # #         phi_ortho = phi_ortho + self.Fock[orb, orb2]*self.phi_prev[orb2][-1]
        # # return -2*self.G_mu[orb]((self.Vnuc + self.J)*self.phi_prev[orb][-1] - self.K[orb] - phi_ortho)
        # phi_ortho = spinor(self.mra, self.Ncomp)  # Initialize as a complex function
        # phi_ortho.setZero()
        # for orb2 in range(self.Norb):
        #     if orb2 != orb:
        #         phi_ortho = phi_ortho + self.Fock[orb, orb2] * self.phi_prev[orb2][-1]
        phi_np1 = spinor(self.mra, self.Ncomp)
        phi_np1_tmp = spinor(self.mra, self.Ncomp)
        phi_np1_tmp = Term1 + Term2 + Term3 + Term4
        for l in range(self.Ncomp):
            if(phi_np1_tmp.compVect[l].real.squaredNorm() > 1e-12):
                # print("PowerIter Orb ok", orb)
                # print("PowerIter, pre Helmholtz, SCF equation", orb, complex_fcn.dot(phi_np1_tmp, phi_np1_tmp))
                phi_np1.compVect[l].real = -2 * self.G_mu[orb](phi_np1_tmp.compVect[l].real)
                # print("PowerIter, post Helmholtz", orb, complex_fcn.dot(phi_np1, phi_np1))
            if(phi_np1_tmp.compVect[l].imag.squaredNorm() > 1e-12):
                phi_np1.compVect[l].imag = -2 * self.G_mu[orb](phi_np1_tmp.compVect[l].imag)
        return phi_np1
    
    #This method sets up then solve the linear system Ac=b for a specific orbital of idex "orb"
    #This operation can be done independantly for each orbital as the ground state orbitals are separated.
    #For the following, the index "orb" will be ommited for the documentation.
    #The matrix elements are defined as A_ij = <phi_history[i]-phi_history[newest]|update_history[j]-update_history[newest]>
    #The vector elements are defined as B_i = <phi_history[i]-phi_history[newest]|update_history[newest]>
    #orb[in]: integer index of the chosen orbital in the list of orbitals "phi_prev"
    #c[out]: numpy array, solution of the linear system
    def setupLinearSystem(self,orb):
        lenHistory = len(self.phi_prev[orb])
        # Compute matrix A
        A = np.zeros((lenHistory-1, lenHistory-1), dtype=complex)
        b = np.zeros(lenHistory-1, dtype=complex)
        for l in range(lenHistory-1):  
            dPhi = self.phi_prev[orb][l] - self.phi_prev[orb][-1]
            b[l] = dPhi.dot( self.f_prev[orb][-1])
            for j in range(lenHistory-1):
                A[l,j] = -dPhi.dot( self.f_prev[orb][j] - self.f_prev[orb][-1])
        #solve Ac = b for c
        c = np.linalg.solve(A, b)
        return c

    #"overlord" method, calls the expandSolution method to update the ground state orbitals until convergence has been achieved.
    #Convergence is defined as the update being smaller than a preset threshold. 
    #Thoughout the run, the values of each orbital's energy, their norm and the update's norm can be printed for reference.
    #thrs[in]: Energy precision threshold in floating point number format, defines when convergence has been achieved.
    #printVal[in]: Boolean value used to toggle on or off the printing of each orbital's norm, update and energy in the terminal.
    #pltShow[in]: Boolean value to toggle on/off the visualisation of the converged orbitals in the x direction
    def scfRun(self, thrs = 1e-3, printVal = False, pltShow = False):
        update = np.ones(self.Norb)
        norm = np.zeros(self.Norb)

        #iteration counter
        i = 0

        # Optimization loop (KAIN) #TODO continuer
        while update.max() > thrs:
            print(f"=============Iteration: {i}")
            i += 1 
            print("F=", self.Fock)
            print("S=",self.computeOverlap())
            self.E_n, norm, update = self.expandSolution()

            for orb in range(self.Norb):
                # this will plot the wavefunction at each iteration
                r_x = np.linspace(-5., 5., 1000)
                phi_n_plt = [self.phi_prev[orb][-1]([x, 0.0, 0.0]) for x in r_x]
                plt.plot(r_x, phi_n_plt) 
                
                if printVal:
                    print(f"Orbital: {orb}    Norm: {norm}    Update: {update}    Energy:{self.E_n}")

        if pltShow:
            plt.show()

    #Utilities
    
    #Method to compute the electron-nuclear contribution to the energy.
    #r[in]: position in 3D space; 3-element vector-like. 
    #[out]: Value of the el-nuc potential at the position "r" in space
    def f_nuc(self, r):   
        out = 0
        #electron-nucleus interaction
        for i in range(self.Nz): 
            out += -self.Z[i]/np.sqrt((r[0]-self.R[i][0])**2 + (r[1]-self.R[i][1])**2 + (r[2]-self.R[i][2])**2) 
        return out
    
    #Method to compute the nuclear-nuclear contribution to the energy.
    #[out]: Value of the nuclear-nuclear potential at the position "r" in space
    def fpp(self):
        #nucleus-nucleus interaction
        out = 0
        for i in range(self.Nz-1):
            for j in range(i+1, self.Nz):
                out += self.Z[i]*self.Z[j]/np.sqrt((self.R[j][0]-self.R[i][0])**2 + (self.R[j][1]-self.R[i][1])**2 + (self.R[j][2]-self.R[i][2])**2)
        return out 

    #Method to compute the overlap matrix whose elements are defined as S_ij = <phi_i|phi_j>, where i,j run over the orbitals, not the history.
    #phi_orth[in]: List of List of function trees. The reason for this peculiar choice is the data format of the orbitals and their history. 
    #This input argument can be used to compute the overlap matrix of some other object. By default, this method computes the overlap matrix of the orbitals. 
    #S[out]: Overlap matrix in numpy array format.
    def computeOverlap(self, phi_orth = None):
        if phi_orth == None:
            phi_orth = self.phi_prev
            length = self.Norb
        else: 
            length = len(phi_orth)
        S = np.zeros((length, length), dtype=complex) #Overlap matrix S_i,j = <Phi^i|Phi^j>
        for i in range(length):
            for j in range(i, length):
                S[i,j] = phi_orth[i][-1].dot(phi_orth[j][-1]) #compute the overlap of the current ([-1]) step
                if i != j:
                    S[j,i] = np.conjugate(S[i,j])
        # print("S", S)
        return S

    #Lödwin orthogonalisation and normalisation
    #phi_orth[in]: List of List of function trees. The reason for this peculiar choice is the data format of the orbitals and their history. 
    #This input argument can be used to orthonormalise some other object.
    #normalise[in]: Boolean value to toggle on/off the normalisation, if unnecessary.
    def orthonormalise(self, phi_in = None, normalise = True):
        if phi_in == None:
            phi_in = self.phi_prev
            length = self.Norb
        else: 
            length = len(phi_in)
        S = self.computeOverlap(phi_in)
        #Diagonalise S to compute S' := S^-1/2 
        eigvals, U = np.linalg.eigh(S) #U is the basis change matrix
        print("orthonorm eigvals=", eigvals)

        s = np.diag(np.power(eigvals, -0.5)) #diagonalised S 
        #Compute s^-1/2
        Sprime = np.dot(U,np.dot(s,np.transpose(U))) # S^-1/2 = U^dagger s^-1/2 U

        #Apply S' to each orbital to obtain a new orthogonal element
        phi_ortho = []
        for i in range(length):
            # phi_tmp =  self.P_eps(utils.Fzero)
            phi_tmp = spinor(self.mra, self.Ncomp)
            phi_tmp.setZero()
            for j in range(length):
            # for j in range(limit[i]):
                phi_tmp = phi_tmp + Sprime[i,j]*phi_in[j][-1]
            if normalise:
                phi_tmp.normalize()
            phi_ortho.append(phi_tmp)
        return phi_ortho


#Child class of scfsolv dedicated to computing linear response orbitals
class scfsolv_1stpert(scfsolv):
    Fock1 : np.ndarray                       #1st order perturbed Fock matrix
    Vpert : vp.FunctionTree                  #1st order perturbed nuclear potential
    J1 : vp.FunctionTree                     #1st order perturbed Coulomb potential
    K1 : list                                #list of 1st order perturbed exchange potential applied to each orbital
    phi_prev1 : list                         #list of all 1st order perturbed orbitals and their KAIN history
    f_prev1 : list                           #list of all 1st order perturbed orbitals updates and their KAIN history
    E1_n : list                              #list of perturbed orbital energies
    pertField : np.ndarray                   #Perturbative field in vector form

    #Constructor of the child class. 
    #Currently only supports copying the attributes from the parent class, scfsolv.
    #Attributes not inherited from the parent are initialised to a default, placeholder value.
    def __init__(self, *args) -> None:
        if type(args[0]) is scfsolv:
            self.__dict__ = args[0].__dict__.copy()
            J1 = args[0].P_eps(utils.Fzero)
        else:
            print("TODO create a non-copy-from-the-parent constructor for scfsolv_1stpert")
            pass 
            # super(B, self).__init__(*args[:2])
            # c = args[2]
        self.J1 = J1
        self.K1 = []
        self.phi_prev1 = []
        self.f_prev1 = []
        self.pertField = np.zeros(3)

    #Initialises the physical properties of the molecule.
    #Perturbative field[in] is a 3-vector specifying the homogeneous perturbative field, currently only supports an electric field 
    def initMolec(self, perturbativeField) -> None:
        self.pertField = perturbativeField
        self.Vpert, mu = self.f_pert()
        #initial guesses can be zero for the perturbed orbitals
        self.phi_prev1 = []
        print(f"=============Initial:")
        for i in range(self.Norb):
            self.phi_prev1.append([self.P_eps(utils.Fzero)]) 
        self.f_prev1 = [[] for i in range(self.Norb)] #list of the corrections at previous steps
        self.compFock()
        # self.printOperators() 

        self.E1_n, update = self.expandSolution()
        
        pert_mat = np.zeros((self.Norb,self.Norb))
        for i in range(len(self.phi_prev)):
            for j in range(len(self.phi_prev)):
                pert_mat[i,j] =  vp.dot(self.phi_prev[i][-1], self.Vpert * self.phi_prev[j][-1])

    #This method is the same as expandSolution but it doesn't include KAIN. It is only useful for debug purposes.
    #Output is the same as the KAIN version.
    def expandSolution_nokain(self):
        #Compute the fock matrix of the system
        self.compFock()
        
        phistory = []
        self.E1_n = []
        update = []
        for orb in range(self.Norb):
            self.E1_n.append(self.Fock1[orb, orb])
            #Compute new power iteration for the Helmholtz operator
            phi_np1 = self.powerIter(orb)     
            #create an alternate history of orbitals which include the power iteration
            phistory.append([phi_np1]) 

        # Orthogonalise the alternate orbital history w.r.t. the unperturbed orbital
        phistory = self.orthogonalise(phistory) 

        for orb in range(self.Norb):

            delta = phistory[orb] - self.phi_prev1[orb][-1]
            self.f_prev1[orb].append(delta)
            #Save new orbital
            self.phi_prev1[orb].append(phistory[orb]) 
            #Correction norm (convergence metric)
            update.append(delta.norm())
            if len(self.phi_prev1[orb]) > self.khist: #deleting oldest element to save memory
                del self.phi_prev1[orb][0]
                del self.f_prev1[orb][0]
        
        # self.printOperators()
        return np.array(self.E1_n),  np.array(update)
    
    #This is the main method of the class; it moves a step forward in the optimisation of the linear response orbitals
    #It makes use of a KAIN accelerator to improves convergence speed.
    #The updated orbitals and history are stored in the class' attributes, and deleted when too old in the history.
    #E_n[out]: Energy level of each orbital in atomic units, outputed in a numpy array
    #update[out]: norm of the difference between the newest and second newest orbital in the history, given in a numpy array.
    def expandSolution(self):
        #Compute the fock matrix of the system
        self.compFock()
        
        phistory = []
        self.E1_n = []
        update = []
        for orb in range(self.Norb):
            self.E1_n.append(self.Fock1[orb, orb])
            #Compute new power iteration for the Helmholtz operator
            phi_np1 = self.powerIter(orb)     
            #create an alternate history of orbitals which include the power iteration
            phistory.append([phi_np1])
        # Orthogonalise the alternate orbital history w.r.t. the unperturbed orbital
        phistory = self.orthogonalise(phistory)

        for orb in range(self.Norb):
            self.f_prev1[orb].append(phistory[orb] - self.phi_prev1[orb][-1])

        c = self.setupLinearSystem_all() 

        for orb in range(self.Norb):
            #Compute the correction delta to the orbitals 
            delta = self.f_prev1[orb][-1] #The c[0]=1 coefficient is implicit here
            for j in range(len(c)):
                delta = delta + c[j]*(self.phi_prev1[orb][j] - self.phi_prev1[orb][-1] + self.f_prev1[orb][j] - self.f_prev1[orb][-1])
            
            #Apply correction
            phi_n = self.phi_prev1[orb][-1]
            phi_n = phi_n + delta
            #Save new orbital
            self.phi_prev1[orb].append(phi_n) 
            #Correction norm (convergence metric)
            update.append(delta.norm())
            if len(self.phi_prev1[orb]) > self.khist: #deleting oldest element to save memory
                del self.phi_prev1[orb][0]
                del self.f_prev1[orb][0]
        
        # self.printOperators()
        return np.array(self.E1_n),  np.array(update)

    #This method sets up (but not solve) the linear system Ac=b for a specific orbital of index "orb"
    #The response orbital are not in general orthogonal to each other, thus the system that needs to be solve is the sum of every systems for each response orbital
    #For the following, the index "orb" will be ommited for the documentation and the "phi" and "update" refer to response orbitals.
    #The matrix elements are defined as A_ij = <phi_history[i]-phi_history[newest]|update_history[j]-update_history[newest]>
    #The vector elements are defined as B_i = <phi_history[i]-phi_history[newest]|update_history[newest]>
    #orb[in]: integer index of the chosen orbital in the list of orbitals "phi_prev"
    #A[out]: numpy array (matrix), matrix A, unsurprisingly 
    #b[out]: numpy array (vector), vector b (!)
    def setupLinearSystem(self,orb):
        lenHistory = len(self.phi_prev1[orb])-1
        # Compute matrix A
        A = np.zeros((lenHistory, lenHistory))
        b = np.zeros(lenHistory)
        for l in range(lenHistory):  
            dPhi = self.phi_prev1[orb][l] - self.phi_prev1[orb][-1]
            b[l] = vp.dot(dPhi, self.f_prev1[orb][-1])
            for j in range(lenHistory):
                A[l,j] = -vp.dot(dPhi, self.f_prev1[orb][j] - self.f_prev1[orb][-1])
        #solve Ac = b for c
        return A, b
    
    #This method adds all the linear systems for every orbitals together and then solves the sum, and returns the solution c
    #As specified in setupLinearSystem's documentation above, this is done because the response orbitals are not seperated, nor orthogonal in general.
    #c[out]: solution of system that is the sum of orbitals' systems
    def setupLinearSystem_all(self):
        lenHistory = len(self.phi_prev1[0])-1
        A = np.zeros((lenHistory, lenHistory))
        b = np.zeros(lenHistory)
        for orb in range(self.Norb):
            Aorb, borb = self.setupLinearSystem(orb)
            A = A + Aorb
            b = b + borb
        #solve Ac = b for c
        c = []
        if b.size > 0:
            c = np.linalg.solve(A, b)
        return c

    #"overlord" method, calls the expandSolution method to update the ground state orbitals until convergence has been achieved.
    #Convergence is defined as the update being smaller than a preset threshold. 
    #Thoughout the run, the values of each orbital's energy (though it might not be useful for response), their norm and the update's norm can be printed for reference.
    #thrs[in]: Energy precision threshold in floating point number format, defines when convergence has been achieved.
    #printVal[in]: Boolean value used to toggle on or off the printing of each orbital's norm, update and energy in the terminal.
    #pltShow[in]: Boolean value to toggle on/off the visualisation of the converged orbitals in the x direction
    def scfRun(self, thrs = 1e-3, printVal = False, pltShow = False):
        update = np.ones(self.Norb)
        norm = np.zeros(self.Norb)
        #iteration counter
        iteration = 1
        # Optimization loop (KAIN) #TODO continuer
        while update.max() > thrs and iteration < 10:
            print(f"=============Iteration: {iteration}")
            iteration += 1 
            if self.khist == 0:
                self.E1_n, update = self.expandSolution_nokain()
            else:
                self.E1_n, update = self.expandSolution()
            for orb in range(self.Norb):
                # this will plot the wavefunction at each iteration
                r_x = np.linspace(-5., 5., 1000)
                phi_n_plt = [self.phi_prev[orb][-1]([x, 0.0, 0.0]) for x in r_x]
                plt.plot(r_x, phi_n_plt) 
                if printVal:
                    print(f"Orbital: {orb}    Norm: {norm}    Update: {update}    Energy:{self.E1_n}")
        if pltShow:
            plt.show()
        
    #===computation of operators===
            
    #This method computes the 1st order perturbed Fock matrix of the molecule
    def compFock(self): 
        self.Fock1 = np.zeros((self.Norb, self.Norb))
        self.K1 = []
        self.J1 = self.computeCoulombOperator()
        for j in range(self.Norb):
            #Compute the potential operator
            self.K1.append(self.computeExchangePotential(j))
            # compute the energy from the orbitals 
            Fphi = self.compFop(j)
            for i in range(self.Norb):
                self.Fock1[j,i] = vp.dot(self.phi_prev[i][-1], Fphi)

    #This method computes the perturbed Fock operator applied to a ground-state (unperturbed) orbital orb
    #orb[in]: integer index of the ground state orbital on which to apply the operator
    def compFop(self, orb): 
        Fphi = self.Vpert*self.phi_prev[orb][-1] + self.J1*self.phi_prev[orb][-1] - self.K1[orb]
        return Fphi
    
    #This method computes a (symmetric) product between one perturbed orbital and an unperturbed one.
    #orb1[in]: integer index of the (un)perturbed orbital on the left side
    #orb2[in]: integer index of the (un)perturbed orbital on the right side
    #rho[out]: product of the two orbitals in function tree (vp.functionTree) format
    def compProduct(self, orb1, orb2 ):
        rho = (self.phi_prev[orb1][-1]*self.phi_prev1[orb2][-1] + self.phi_prev1[orb1][-1]*self.phi_prev[orb2][-1])
        return rho

    #This method computes the 1st order perturbed Coulomb operator
    #[out]: Coulomb operator in function tree (vp.functionTree) format
    def computeCoulombOperator(self): 
        PNbr = 4*np.pi*self.compProduct(0,0)
        for orb in range(1, self.Norb):
            PNbr = PNbr + 4*np.pi*self.compProduct(orb,orb)
        return self.Pois(2*PNbr) #factor of 2 because we sum over the number of orbitals, not electrons
    
    #This method computes the 1st order perturbed exchange operator applied to an unperturbed orbital of index "idx"
    # idx[in]: integer index of the chosen orbital in the list of orbitals "phi_prev"
    # K_idx[out]: function tree (vp.FunctionTree) representation of the operator*orbital 
    def computeExchangePotential(self, idx):
        K_idx = self.phi_prev[0][-1]*self.Pois(4*np.pi*self.phi_prev1[0][-1]*self.phi_prev[idx][-1]) + self.phi_prev1[0][-1]*self.Pois(4*np.pi*self.phi_prev[0][-1]*self.phi_prev[idx][-1])
        for j in range(1, self.Norb): #summing over occupied orbitals
            K_idx = K_idx + self.phi_prev[j][-1]*self.Pois(4*np.pi*self.phi_prev1[j][-1]*self.phi_prev[idx][-1]) + self.phi_prev1[j][-1]*self.Pois(4*np.pi*self.phi_prev[j][-1]*self.phi_prev[idx][-1])
        return K_idx 
    
    #This method computes the UNperturbed exchange operator applied to a 1st order perturbed orbital of index "idx"
    # idx[in]: integer index of the chosen orbital in the list of orbitals "phi_prev1"
    # K_idx[out]: function tree (vp.FunctionTree) representation of the operator*orbital 
    def computeUnperturbedExchangePotential(self, idx):
        K_idx = self.phi_prev[0][-1]*self.Pois(4*np.pi*self.phi_prev[0][-1]*self.phi_prev1[idx][-1])
        for j in range(1, self.Norb): #summing over occupied orbitals
            K_idx = K_idx + self.phi_prev[j][-1]*self.Pois(4*np.pi*self.phi_prev[j][-1]*self.phi_prev1[idx][-1])
        return K_idx 

    #This method executes one SCF "power iteration", i.e. one application of the Helmholtz operator to the 1st order perturbed SCF equation
    #orb[in]: integer index of the chosen orbital in the list of orbitals "phi_prev"
    #[out]: function tree (vp.FunctionTree) representation of the updated orbital with index "orb" 
    def powerIter(self, orb): #Devrait suivre la méthode qu'utilise MRChem plus précisément
        phi_ortho = self.P_eps(utils.Fzero)
        #Take into account the non-canonical constraint Sum_{j≠i} F^0_ij|phi^1_{j}>
        for j in range(self.Norb): 
            if j != orb:
                phi_ortho = phi_ortho + self.Fock[orb, j]*self.phi_prev1[j][-1] 
        Fphi = self.orthogonalise([[self.compFop(orb)]])[0] #the 0 index here is necessary because compFop returns a one-element list of fctTree

        #Compute K^0 |phi^1>
        K0phi1 = self.computeUnperturbedExchangePotential(orb)
        return -2*self.G_mu[orb](self.Vnuc*self.phi_prev1[orb][-1] + self.J*self.phi_prev1[orb][-1] - K0phi1 - phi_ortho + Fphi)

    
    #Dipole moment and polarisability computation
    #drct[in]: direction of the dipole moment (x=0, y=1, z=2)
    #nuclei_width[in]: smearing of the gaussians representing the nuclei, by default this is set as the precision of the grid to simulate point charges.
    #[out]: tuple containing 3 elements (all of them in vp.functionTree format), listed below: 
    # 1) Dipole moment in the selected direction 
    # 2) Electronic contribution to the dipole moment
    # 3) Nuclear contribution to the dipole moment (constant throughout the SCF process)
    def compDiMo(self, drct = 0, nuclei_width = 0): #computes the dipole moment operator
        #The electron contribution to the dipole moment is only a position operator
        r_i = self.P_eps(lambda r : utils.Flin(r, drct))
        electronContrib = -1*r_i #charge of an e- is -1

        #Computing nuclear contribution (Greatly inspired by Gabriel's 'constructChargeDensity' function in MRPyCM)
        nucContrib = vp.GaussExp() #creates a sum of Gaussian, treated like a list/array
        if nuclei_width <= 0: 
            nuclei_width = self.prec #The standard deviation of the Gaussian functions are of the same order of magnitude as the precision, as they effectively represent point charges
        for nuc in range(len(self.Z)):
            norm_factor = (nuclei_width / np.pi)**(3./2.) #Gaussian normalisation factor. 'nuclei_width' is the exponent factor of the Gaussians 
            nucContrib.append(vp.GaussFunc(exp=nuclei_width, coef=norm_factor*self.Z[nuc], pos=self.R[nuc], pow=[0,0,0]))
        return electronContrib + self.P_eps(nucContrib), electronContrib, nucContrib

    #Utilities

    #This function computes the perturbation operator
    #It computes -F \cdot \mu 
    #[out]: tuple containing the perturbation operator in vp.functionTree format and and a tuple containing the output of compDiMo.
    def f_pert(self) -> tuple:
        #The perturbative field contribution to the energy is of the form $-\vec{\mu}\cdot\vec{\epsilon}$ 
        out = self.P_eps(utils.Fzero) #dipole moment
        #This loop is basically a scalar product between epsilon (represented by self.pertField) and the dipole moment mu
        mu = [0 for i in range(3)]
        for direction in range(3): #3 directions: x,y and z
            # computing the current component of the dipole moment
            mu[direction], muEl, muNuc = self.compDiMo(drct=direction, nuclei_width=0)
            # computing the scalar product
            out = out + self.pertField[direction] * mu[direction] / np.linalg.norm(self.pertField)
        return out, (mu, muEl, muNuc)
    

    #Computes the overlap between phi_in and the unperturbed orbitals, defined as S_ij = <phi_i|phi_in_j>, where i,j run over the orbitals, not the history.
    #phi_in[in]: List of List of function trees. The reason for this peculiar choice is the data format of the orbitals and their history. 
    #This input argument can be used to compute the overlap matrix of some other object. By default, phi_in contains the 1st order perturbed orbitals. 
    #S[out]: Overlap matrix in numpy array format.
    def computeOverlap(self, phi_in = None):
        if phi_in == None:
            phi_in = self.phi_prev1
        S = np.zeros((self.Norb, self.Norb)) #Overlap matrix S_i,j = <Phi^i|Phi^j>
        for i in range(len(phi_in)):
            for j in range(self.Norb):
                S[i,j] = vp.dot(self.phi_prev[j][-1], phi_in[i][-1]) #compute the overlap of the current ([-1]) step
        return S
    
    #Gram-Schmidt orthogonalisation w.r.t. the unperturbed orbitals, no normalisation
    #phi_in[in]: List of List of function trees, refer to computeOverlap's doc string to know slightly more.
    #phi_ortho[out]: list (NOT list of list, because of bad coding) of the orthogonalised orbitals (in vp.functionTree format)
    def orthogonalise(self, phi_in = None): 
        if phi_in == None:
            phi_in = self.phi_prev1
        S = self.computeOverlap(phi_in)
        #Apply S' to each orbital to obtain a new orthogonal element
        phi_ortho = []
        for i in range(len(phi_in)):
            phi_tmp =  phi_in[i][-1]
            for j in range(self.Norb):
                phi_tmp = phi_tmp - S[i,j]*self.phi_prev[j][-1]
            phi_ortho.append(phi_tmp)
        return phi_ortho

    #Prints the operators' matrices in the basis of the unperturbed orbitals in the terminal. 
    #The best debugger in existance. 
    def printOperators(self):
        self.compFock()
        coulomb = np.zeros((self.Norb,self.Norb))
        exchange = np.zeros((self.Norb,self.Norb))
        for orb1 in range(self.Norb):
            J1phi0 = self.J1*self.phi_prev[orb1][-1]
            for orb2 in range(self.Norb):
                coulomb[orb1,orb2] = vp.dot(self.phi_prev[orb2][-1], J1phi0) 
                exchange[orb1,orb2] = vp.dot(self.phi_prev[orb2][-1], self.K1[orb1])
                print("correlation", vp.dot(self.phi_prev[orb2][-1], self.phi_prev[orb1][-1]), vp.dot(self.phi_prev[orb1][-1], self.phi_prev[orb2][-1]))
        print("Fock", self.Fock1)
        print("Coulomb", coulomb)
        print("Exchange", exchange)
