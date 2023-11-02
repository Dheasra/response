from vampyr import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt
# from copy import deepcopy

# import KAIN
import utils

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



    def __init__(self, prec, khist, lgdrOrder=6, sizeScale=-4, nboxes=2, scling=1.0) -> None:
        self.world = vp.BoundingBox(corner=[-1]*3, nboxes=[nboxes]*3, scaling=[scling]*3, scale= sizeScale)
        self.mra = vp.MultiResolutionAnalysis(order=lgdrOrder, box=self.world)
        self.prec = prec
        self.P_eps = vp.ScalingProjector(self.mra, self.prec) 
        self.Pois = vp.PoissonOperator(self.mra, self.prec)
        #Derivative operator
        self.D = vp.ABGVDerivative(self.mra, a=0.5, b=0.5)
        #Physical properties (placeholder values)
        self.Norb = 1
        self.Fock = np.zeros((self.Norb, self.Norb))
        self.J = self.P_eps(utils.Fzero)
        self.Vnuc = self.P_eps(utils.Fzero)
        self.K = []
        self.R = []
        self.Z = []
        self.Nz = 1
        self.E_pp = 0.0
        #Accelerator
        self.khist = khist
        self.phi_prev = []
        self.f_prev = []
        # print(self.world)

    def init_molec(self, No,  pos, Z, init_g_dir) -> None:
        self.Nz = len(Z)
        self.Norb = No
        self.R = pos
        self.Z = Z
        self.Vnuc = self.P_eps(lambda r : self.f_nuc(r))
        #initial guesses provided by mrchem
        self.phi_prev = []
        for i in range(self.Norb):
            ftree = vp.FunctionTree(self.mra)
            print(f"{init_g_dir}phi_p_scf_idx_{i}_re")
            ftree.loadTree(f"{init_g_dir}phi_p_scf_idx_{i}_re") 
            self.phi_prev.append([ftree])
        self.f_prev = [[] for i in range(self.Norb)] #list of the corrections at previous steps
        #Compute the Fock matrix and potential operators 
        self.compFock()
        # print("squalalala",self.Fock)
        #Energies of each orbital
        self.E_n = []
        #internal nuclear energy
        self.E_pp = self.fpp()
        # Prepare Helmholtz operators
        self.G_mu = []
        # J = computeCoulombPot(phi_prev)
        for i in range(self.Norb):
            self.E_n.append(self.Fock[i,i])
            # self.E_n.append(2*F[i]+E_pp)
            mu = np.sqrt(-2*self.E_n[i])
            # mu = 1 #E = -0.5 analytical solution
            self.G_mu.append(vp.HelmholtzOperator(self.mra, mu, self.prec))  # Initalize the operator
        for orb in range(self.Norb):
            #First establish a history (at least one step) of corrections to the orbital with standard iterations with Helmholtz operator to create
            # Apply Helmholtz operator to obtain phi_np1 #5
            phi_np1 = self.powerIter(orb)
            # Compute update = ||phi^{n+1} - phi^{n}|| #6
            self.f_prev[orb].append(phi_np1 - self.phi_prev[orb][-1])
            phi_np1.normalize()
            self.phi_prev[orb].append(phi_np1)
        #Orthonormalise orbitals since they are molecular orbitals
        phi_ortho = self.orthonormalise()
        for orb in range(self.Norb): #Mandatory loop due to questionable data format choice.
            self.phi_prev[orb][-1] = phi_ortho[orb]

    # def compOperators(self): #Compute operators 


    #computation of operators
    def compFock(self): 
        self.Fock = np.zeros((self.Norb, self.Norb))
        self.K = []
        self.J = self.computeCoulombPot()
        for j in range(self.Norb):
            #Compute the potential operator
            self.K.append(self.computeExchangePotential(j))
            # V = Vnuc
            # compute the energy from the orbitals 
            Fphi = self.compFop(j)
            for i in range(self.Norb):
                self.Fock[i,j] = vp.dot(self.phi_prev[i][-1], Fphi)

    def compFop(self, orb): #Computes the Fock operator applied to an orbital orb #TODO: modifier ça pour que ça rentre dans compFock
        Tphi = -0.5*(self.D(self.D(self.phi_prev[orb][-1], 0), 0) + self.D(self.D(self.phi_prev[orb][-1], 1), 1) + self.D(self.D(self.phi_prev[orb][-1], 2), 2)) #Laplacian of the orbitals
        # Fphi = Tphi + self.Vnuc[orderF]*self.phi_prev[orb][-1] + self.J[orderF]*self.phi_prev[orb][-1] - self.K[orb][orderF]
        Fphi = Tphi + self.Vnuc*self.phi_prev[orb][-1] + self.J*self.phi_prev[orb][-1] - self.K[orb]
        return Fphi

    def compScalarPrdt(self, orb1, orb2): #Computes the scalar product between an electron in orbital orb1 and another in orbital orb2
                                          #NOTE: to compute the density, a factor 2 will be lacking 
        rho = self.phi_prev[orb1][-1]*self.phi_prev[orb2][-1]
        return rho

    def computeCoulombPot(self): 
        PNbr = 4*np.pi*self.compScalarPrdt(0, 0)
        for orb in range(1, self.Norb):
            PNbr = PNbr + 4*np.pi*self.compScalarPrdt(orb, orb)
        return self.Pois(2*PNbr) #factor of 2 because we sum over the number of orbitals, not electrons

    def computeExchangePotential(self, idx):
        K = self.phi_prev[0][-1]*self.Pois(4*np.pi*self.compScalarPrdt(0, idx))
        for j in range(1, self.Norb):
            K = K + self.phi_prev[j][-1]*self.Pois(4*np.pi*self.compScalarPrdt(j, idx))
        return K 
    
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
            c = self.setuplinearsystem(orb)
            #Compute the correction delta to the orbitals 
            delta = self.f_prev[orb][-1]
            for j in range(len(self.phi_prev[orb])-1):
                delta = delta + c[j]*(self.phi_prev[orb][j] - self.phi_prev[orb][-1] + self.f_prev[orb][j] - self.f_prev[orb][-1])
            #Apply correction
            phi_n = self.phi_prev[orb][-1]
            phi_n = phi_n + delta
            #Normalize
            norm.append(phi_n.norm())
            phi_n.normalize()
            #Save new orbital
            self.phi_prev[orb].append(phi_n)
            #Correction norm (convergence metric)
            update.append(delta.norm())
            if len(self.phi_prev[orb]) > self.khist: #deleting oldest element to save memory
                del self.phi_prev[orb][0]
                del self.f_prev[orb][0]
        return np.array(self.E_n), np.array(norm), np.array(update)
    
    def powerIter(self, orb):
        #Compute phi_ortho := Sum_{j!=i} F_ij*phi_j 
        phi_ortho = self.P_eps(utils.Fzero) 
        for orb2 in range(self.Norb):
            #Compute off-diagonal Fock matrix elements
            if orb2 != orb:
                phi_ortho = phi_ortho + self.Fock[orb, orb2]*self.phi_prev[orb2][-1]
        return -2*self.G_mu[orb]((self.Vnuc + self.J)*self.phi_prev[orb][-1] - self.K[orb] - phi_ortho)
    
    def setuplinearsystem(self,orb):
        lenHistory = len(self.phi_prev[orb])
        # Compute matrix A
        A = np.zeros((lenHistory-1, lenHistory-1))
        b = np.zeros(lenHistory-1)
        for l in range(lenHistory-1):  
            dPhi = self.phi_prev[orb][l] - self.phi_prev[orb][-1]
            b[l] = vp.dot(dPhi, self.f_prev[orb][-1])
            for j in range(lenHistory-1):
                A[l,j] = -vp.dot(dPhi, self.f_prev[orb][j] - self.f_prev[orb][-1])
        #solve Ac = b for c
        c = np.linalg.solve(A, b)
        return c

    def scfRun(self, thrs = 1e-3, printVal = False, pltShow = False):
        update = np.ones(self.Norb)
        norm = np.zeros(self.Norb)

        #iteration counter
        i = 0

        # Optimization loop (KAIN) #TODO continuer
        while update.max() > thrs:
            print(f"=============Iteration: {i}")
            i += 1 
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
    def f_nuc(self, r):   
        out = 0
        #electron-nucleus interaction
        for i in range(self.Nz): 
            out += -self.Z[i]/np.sqrt((r[0]-self.R[i][0])**2 + (r[1]-self.R[i][1])**2 + (r[2]-self.R[i][2])**2) 
        return out
    
    def fpp(self):
        #nucleus-nucleus interaction
        out = 0
        for i in range(self.Nz-1):
            for j in range(i+1, self.Nz):
                out += self.Z[i]*self.Z[j]/np.sqrt((self.R[j][0]-self.R[i][0])**2 + (self.R[j][1]-self.R[i][1])**2 + (self.R[j][2]-self.R[i][2])**2)
        return out 

    def computeOverlap(self, phi_orth = None):
        # if phi_orth == None:
        #     phi_orth = self.phi_prev
        if phi_orth == None:
            phi_orth = self.phi_prev
            length = self.Norb
        else: 
            length = len(phi_orth)
        S = np.zeros((length, length)) #Overlap matrix S_i,j = <Phi^i|Phi^j>
        for i in range(length):
            for j in range(i, length):
                S[i,j] = vp.dot(phi_orth[i][-1], phi_orth[j][-1]) #compute the overlap of the current ([-1]) step
                if i != j:
                    S[j,i] = np.conjugate(S[i,j])
        print(S)
        return S

    def orthonormalise(self, phi_in = None, normalise = True): #Lödwin orthogonalisation and normalisation
        if phi_in == None:
            phi_in = self.phi_prev
            length = self.Norb
        else: 
            length = len(phi_in)
        S = self.computeOverlap(phi_in)
        #Diagonalise S to compute S' := S^-1/2 
        eigvals, U = np.linalg.eigh(S) #U is the basis change matrix

        s = np.diag(np.power(eigvals, -0.5)) #diagonalised S 
        #Compute s^-1/2
        Sprime = np.dot(U,np.dot(s,np.transpose(U))) # S^-1/2 = U^dagger s^-1/2 U

        print("U=", U)
        #Apply S' to each orbital to obtain a new orthogonal element
        phi_ortho = []
        for i in range(length):
            phi_tmp =  self.P_eps(utils.Fzero)
            for j in range(length):
            # for j in range(limit[i]):
                phi_tmp = phi_tmp + Sprime[i,j]*phi_in[j][-1]
            if normalise:
                phi_tmp.normalize()
            phi_ortho.append(phi_tmp)
        return phi_ortho


class scfsolv_1stpert(scfsolv):
    Fock1 : np.ndarray                       #1st order perturbed Fock matrix
    Vpert : vp.FunctionTree                  #1st order perturbed nuclear potential
    J1 : vp.FunctionTree                     #1st order perturbed Coulomb potential
    K1 : list                                #list of 1st order perturbed exchange potential applied to each orbital
    phi_prev1 : list                         #list of all 1st order perturbed orbitals and their KAIN history
    f_prev1 : list                           #list of all 1st order perturbed orbitals updates and their KAIN history
    E1_n : list                              #list of perturbed orbital energies
    pertField : np.ndarray                   #Perturbative field in vector form
    # mu : np.ndarray                          #Dipole moment

    # def __init__(self, prec, khist, lgdrOrder=6, sizeScale=-4, nboxes=2, scling=1.0) -> None: 
    #     super().__init__(prec, khist, lgdrOrder, sizeScale, nboxes, scling)
    #     self.J1 = super().P_eps(utils.Fzero)
    #     self.K1 = []
    #     self.phi_prev1 = []
    #     self.f_prev1 = []

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

    def init_molec(self, perturbativeField) -> None:
        self.pertField = perturbativeField
        # print("init_molec start")
        # self.Vpert = self.P_eps(lambda r : self.f_pert(r)) 
        self.Vpert, mu = self.f_pert()
        # print("Init_prout")
        #initial guesses can be zero for the perturbed orbitals
        self.phi_prev1 = []
        for i in range(self.Norb):
            self.phi_prev1.append([self.P_eps(utils.Fzero)]) #TODO: faire en sorte que les perturbations soient dans les trois directions --> rajouer une couche de liste
        self.f_prev1 = [[] for i in range(self.Norb)] #list of the corrections at previous steps
        # print("passed init orb")
        #Compute the Fock matrix and potential operators 
        self.compFock()                                                         
        # print("squalalala",self.Fock)
        #Energies of each orbital
        self.E1_n = []
        # #internal nuclear energy
        # self.E_pp = super().fpp()
        for i in range(self.Norb):
            self.E1_n.append(self.Fock[i,i])
            # mu = np.sqrt(-2*super().E_n[i])
            # mu = 1 #E = -0.5 analytical solution
            # super().G_mu.append(vp.HelmholtzOperator(super().mra, mu, super().prec))  # Initalize the operator
        for orb in range(self.Norb):
            #First establish a history (at least one step) of corrections to the orbital with standard iterations with Helmholtz operator to create
            # Apply Helmholtz operator to obtain phi_np1 #5
            phi_np1 = self.powerIter(orb)
            # Compute update = ||phi^{n+1} - phi^{n}|| #6
            self.f_prev1[orb].append(phi_np1 - self.phi_prev1[orb][-1])
            # phi_np1.normalize()
            self.phi_prev1[orb].append(phi_np1)
        # #Orthonormalise orbitals since they are molecular orbitals
        phi_ortho = self.orthogonalise()
        for orb in range(self.Norb):
            self.phi_prev1[orb][-1] = phi_ortho[orb]
        print("TEST init_molec")
        for i in range(len(self.phi_prev)):
            for j in range(len(self.phi_prev1)):
                print(f"Test: {i}, {j}", vp.dot(self.phi_prev[i][-1], self.phi_prev1[j][-1]))
    
    def expandSolution(self):
        # Orthogonalise the alternate orbital history w.r.t. the unperturbed orbital in case they aren't yet
        phi_ortho = self.orthogonalise()
        for orb in range(self.Norb): #Mandatory loop due to questionable data format choice.
            self.phi_prev1[orb][-1] = phi_ortho[orb]

        #Compute the fock matrix of the system
        self.compFock()
        
        phistory = []
        self.E1_n = []
        # norm = []
        update = []
        for orb in range(self.Norb):
            self.E1_n.append(self.Fock1[orb, orb])
            #Compute new power iteration for the Helmholtz operator
            phi_np1 = self.powerIter(orb)        
            #create an alternate history of orbitals which include the power iteration
            phistory.append([phi_np1])

        # Orthogonalise the alternate orbital history w.r.t. the unperturbed orbital
        phistory = self.orthogonalise(phistory)
        # print("TEST expandSol")
        # for i in range(len(self.phi_prev)):
        #     for j in range(len(self.phi_prev1)):
        #         print(f"Test: {i}, {j}", vp.dot(self.phi_prev[i][-1], self.phi_prev1[j][-1]))

        # phi_n_ortho = []
        for orb in range(self.Norb):
            self.f_prev1[orb].append(phistory[orb] - self.phi_prev1[orb][-1])
            #Setup and solve the linear system Ac=b
            c = self.setuplinearsystem(orb)
            #Compute the correction delta to the orbitals 
            delta = self.f_prev[orb][-1]
            for j in range(len(self.phi_prev1[orb])-1):
                delta = delta + c[j]*(self.phi_prev1[orb][j] - self.phi_prev1[orb][-1] + self.f_prev1[orb][j] - self.f_prev1[orb][-1])
            #Apply correction
            phi_n = self.phi_prev1[orb][-1]
            phi_n = phi_n + delta
            # phi_n_ortho.append([self.phi_prev1[orb][-1] + delta]) #wrong
        # phi_n = self.orthogonalise(phi_n_ortho)  #wrong
        # for orb in range(self.Norb): #wrong
            # #Normalize
            # norm.append(phi_n.norm())
            # phi_n.normalize()
            #Save new orbital
            self.phi_prev1[orb].append(phi_n)
            #Correction norm (convergence metric)
            update.append(delta.norm())
            if len(self.phi_prev1[orb]) > self.khist: #deleting oldest element to save memory
                del self.phi_prev1[orb][0]
                del self.f_prev1[orb][0]
        return np.array(self.E1_n),  np.array(update)

    def setuplinearsystem(self,orb):
        lenHistory = len(self.phi_prev1[orb])
        # Compute matrix A
        A = np.zeros((lenHistory-1, lenHistory-1))
        b = np.zeros(lenHistory-1)
        for l in range(lenHistory-1):  
            dPhi = self.phi_prev1[orb][l] - self.phi_prev1[orb][-1]
            b[l] = vp.dot(dPhi, self.f_prev1[orb][-1])
            for j in range(lenHistory-1):
                A[l,j] = -vp.dot(dPhi, self.f_prev1[orb][j] - self.f_prev1[orb][-1])
        #solve Ac = b for c
        c = np.linalg.solve(A, b)
        return c

    def scfRun(self, thrs = 1e-3, printVal = False, pltShow = False):
        update = np.ones(self.Norb)
        norm = np.zeros(self.Norb)
        #iteration counter
        iteration = 0
        # Optimization loop (KAIN) #TODO continuer
        while update.max() > thrs:
            print(f"=============Iteration: {iteration}")
            iteration += 1 
            self.E1_n, update = self.expandSolution()
            # print("TEST run")
            # for i in range(len(self.phi_prev)):
            #     for j in range(len(self.phi_prev1)):
            #         print(f"Test: {i}, {j}", vp.dot(self.phi_prev[i][-1], self.phi_prev1[j][-1]))
            for orb in range(self.Norb):
                # this will plot the wavefunction at each iteration
                r_x = np.linspace(-5., 5., 1000)
                phi_n_plt = [self.phi_prev[orb][-1]([x, 0.0, 0.0]) for x in r_x]
                plt.plot(r_x, phi_n_plt) 
                if printVal:
                    print(f"Orbital: {orb}    Norm: {norm}    Update: {update}    Energy:{self.E1_n}")
        if pltShow:
            plt.show()
        
    #computation of operators
    def compFock(self): 
        self.Fock1 = np.zeros((self.Norb, self.Norb))
        self.K1 = []
        self.J1 = self.computeCoulombPot()
        for j in range(self.Norb):
            #Compute the potential operator
            self.K1.append(self.computeExchangePotential(j))
            # V = Vnuc
            # compute the energy from the orbitals 
            Fphi = self.compFop(j)
            for i in range(self.Norb):
                self.Fock1[i,j] = vp.dot(self.phi_prev[i][-1], Fphi)

    def compFop(self, orb): #Computes the Fock operator applied to an orbital orb #TODO: modifier ça pour que ça rentre dans compFock
        Tphi = -0.5*(self.D(self.D(self.phi_prev[orb][-1], 0), 0) + self.D(self.D(self.phi_prev[orb][-1], 1), 1) + self.D(self.D(self.phi_prev[orb][-1], 2), 2)) #Laplacian of the orbitals
        # Fphi = Tphi + self.Vnuc[orderF]*self.phi_prev[orb][-1] + self.J[orderF]*self.phi_prev[orb][-1] - self.K[orb][orderF]
        Fphi = Tphi + self.Vnuc*self.phi_prev[orb][-1] + self.J1*self.phi_prev[orb][-1] - self.K1[orb]
        return Fphi
    
    def compScalarPrdt(self, orb1, orb2 ):
        # print("comp Rho", orb1, orb2)
        # print(len(self.phi_prev), len(self.phi_prev1))
        rho = self.phi_prev[orb1][-1]*self.phi_prev1[orb2][-1] + self.phi_prev1[orb1][-1]*self.phi_prev[orb2][-1]
        return 2*rho

    def computeCoulombPot(self): 
        PNbr = 4*np.pi*self.compScalarPrdt(0,0)
        for orb in range(1, self.Norb):
            PNbr = PNbr + 4*np.pi*self.compScalarPrdt(orb,orb)
        return self.Pois(2*PNbr) #factor of 2 because we sum over the number of orbitals, not electrons

    def computeExchangePotential(self, idx):
        K = self.phi_prev[0][-1]*self.Pois(4*np.pi*self.compScalarPrdt(0,idx))
        for j in range(1, self.Norb):
            K = K + self.phi_prev[j][-1]*self.Pois(4*np.pi*self.compScalarPrdt(j,idx))
        return K 
    
    def computeUnperturbedExchangePotential(self, idx):
        K = self.phi_prev1[0][-1]*self.Pois(4*np.pi*super().compScalarPrdt(0,idx))
        for j in range(1, self.Norb):
            K = K + self.phi_prev[j][-1]*self.Pois(4*np.pi*super().compScalarPrdt(j,idx))
        return K 

    def powerIter_old(self, orb): #TODO: Probablement un problème dans le (1-rho0) 
        # if order == 1:
        # print("TODODODODODODODODO")
        Fphi = self.compFop(orb) 
        # self.Vpert = self.f_pert()
        # rho0 = self.compScalarPrdt(0, 0, 0)
        # for o in range(1,self.Norb):
        #     rho0 = rho0 + self.compScalarPrdt(o, o, 0)
        phi_ortho = self.P_eps(utils.Fzero)
        #Compute the orthonormalisation constraint Sum_{j≠i} F^0_ij|phi^1_{j}>
        for j in range(self.Norb): 
            # phi_ortho = phi_ortho + self.Fock[0][orb, j]*self.phi_prev[j][-2][1] 
            phi_ortho = phi_ortho + self.Fock[orb, j]*self.phi_prev1[j][-1] 
        #Compute the term \hat{rho}^0*\hat{F}^1|phi^0_i>
        rhoFphi = self.P_eps(utils.Fzero)
        for j in range(self.Norb):
            rhoFphi += self.Fock1[j, orb] * self.phi_prev[j][-1]
        # print(len(self.G_mu), len(self.phi_prev1), len(self.K))

        #Compute K^0 |phi^1>
        K0phi1 = self.computeUnperturbedExchangePotential(orb)
        print("Test perturbed space", orb, vp.dot(Fphi - rhoFphi, self.phi_prev[orb][-1]))
        return -2*self.G_mu[orb](self.Vpert*self.phi_prev1[orb][-1] + self.J*self.phi_prev1[orb][-1] - K0phi1 - phi_ortho + Fphi - rhoFphi) 
    
    def powerIter(self, orb): #Devrait suivre la méthode qu'utilise MRChem plus précisément
        # if order == 1:
        # print("TODODODODODODODODO")
        # self.Vpert = self.f_pert()
        # rho0 = self.compScalarPrdt(0, 0, 0)
        # for o in range(1,self.Norb):
        #     rho0 = rho0 + self.compScalarPrdt(o, o, 0)
        phi_ortho = self.P_eps(utils.Fzero)
        #Compute the orthonormalisation constraint Sum_{j≠i} F^0_ij|phi^1_{j}>
        for j in range(self.Norb): 
            # phi_ortho = phi_ortho + self.Fock[0][orb, j]*self.phi_prev[j][-2][1] 
            phi_ortho = phi_ortho + self.Fock[orb, j]*self.phi_prev1[j][-1] 
        Fphi = self.orthogonalise([[self.compFop(orb)]])[0] 
        # #Compute the term \hat{rho}^0*\hat{F}^1|phi^0_i>
        # rhoFphi = self.P_eps(utils.Fzero)
        # for j in range(self.Norb):
        #     rhoFphi += self.Fock1[j, orb] * self.phi_prev[j][-1]
        # print(len(self.G_mu), len(self.phi_prev1), len(self.K))

        #Compute K^0 |phi^1>
        K0phi1 = self.computeUnperturbedExchangePotential(orb)
        print("Test projection to orthogonal space", orb, vp.dot(Fphi, self.phi_prev[orb][-1]))
        return -2*self.G_mu[orb](self.Vpert*self.phi_prev1[orb][-1] + self.J*self.phi_prev1[orb][-1] - K0phi1 - phi_ortho + Fphi) 
    
    #Dipole moment and polarisability computation
    def compDiMo(self, drct = 0):
        # print("comp Dipole Moment")
        rho0 = 2*super().compScalarPrdt(0, 0) #Factor 2 because each orbital contains 2 electrons (closed shelldundun)
        for orb in range(1, self.Norb):
            rho0 += 2*super().compScalarPrdt(orb, orb)
        # print("Pre projection")
        r_i = self.P_eps(lambda r : utils.Flin(r, drct))
        # print("ATTENTION: TEST/ CHANGE Fx BACK TO Flin IN compDiMo")
        # r_i = self.P_eps(utils.Fx)
        integrand = rho0*r_i
        integrand.crop(self.prec)
        electronContrib = -1*(integrand.integrate())
        #Computing nuclear contribution 
        nucContrib = 0 
        for nuc in range(len(self.Z)):
            nucContrib += self.Z[nuc]*self.R[nuc][drct]
        return electronContrib + nucContrib, electronContrib, nucContrib

    #Utilities
    def f_pert(self) -> tuple:   #TODO: Corriger le fait que ça retourne un float plutôt qu'un functiontree
        #The perturbative field contribution to the energy is of the form $-\vec{\mu}\cdot\vec{\epsilon}$ 
        out = 0 #dipole moment
        #This loop is basically a scalar product between epsilon (represented by self.pertField) and the dipole moment mu
        mu = np.zeros(3)
        for direction in range(3): #3 directions: x,y and z
            # computing the current component of the dipole moment
            mu[direction], muEl, muNuc = self.compDiMo(direction)
            print(f"mu_{direction} = ", mu[direction])
            # computing the scalar product
            out += self.pertField[direction] * mu[direction]
        return out, (mu, muEl, muNuc)
  
    def computeOverlap(self, phi_in = None): #Computes the overlap between phi_in and the unperturbed orbitals
        # return super().computeOverlap(phi_orth)
        if phi_in == None:
            phi_in = self.phi_prev1
        S = np.zeros((self.Norb, self.Norb)) #Overlap matrix S_i,j = <Phi^i|Phi^j>
        for i in range(len(phi_in)):
            for j in range(self.Norb):
                S[i,j] = vp.dot(self.phi_prev[j][-1], phi_in[i][-1]) #compute the overlap of the current ([-1]) step
        return S
    
    def orthogonalise(self, phi_in = None):
        if phi_in == None:
            phi_in = self.phi_prev1
        S = self.computeOverlap(phi_in)
        #Apply S' to each orbital to obtain a new orthogonal element
        phi_ortho = []
        for i in range(len(phi_in)):
            # phi_tmp =  self.P_eps(utils.Fzero)
            phi_tmp =  phi_in[i][-1]
            for j in range(self.Norb):
            # for j in range(limit[i]):
                phi_tmp = phi_tmp - S[i,j]*self.phi_prev[j][-1]
            # if normalise:
            #     phi_tmp.normalize()
            phi_ortho.append(phi_tmp)
        return phi_ortho

    # def print_operators(self):
    #     for orb1 in range(self.Norb):
    #         for orb2 in range(self.Norb):
