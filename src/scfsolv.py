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
            print("past power iter")
            # Compute update = ||phi^{n+1} - phi^{n}|| #6
            self.f_prev[orb].append(phi_np1 - self.phi_prev[orb][-1])
            phi_np1.normalize()
            self.phi_prev[orb].append(phi_np1)
        #Orthonormalise orbitals since they are molecular orbitals
        phi_ortho = self.orthonormalise()
        for orb in range(self.Norb): #Mandatory loop due to questionable data format choice.
            self.phi_prev[orb][-1] = phi_ortho[orb]

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

    def compRho(self, orb1, orb2):
        rho = self.phi_prev[orb1][-1]*self.phi_prev[orb2][-1]
        return rho

    def computeCoulombPot(self): 
        PNbr = 4*np.pi*self.compRho(0, 0)
        for orb in range(1, self.Norb):
            PNbr = PNbr + 4*np.pi*self.compRho(orb, orb)
        return self.Pois(2*PNbr) #factor of 2 because we sum over the number of orbitals, not electrons

    def computeExchangePotential(self, idx):
        K = self.phi_prev[0][-1]*self.Pois(4*np.pi*self.compRho(0, idx))
        for j in range(1, self.Norb):
            K = K + self.phi_prev[j][-1]*self.Pois(4*np.pi*self.compRho(j, idx))
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
        # if order == 1:
        #     # print("TODODODODODODODODO")
        #     Fphi = self.compFop(orb, 1, 0)
        #     # rho0 = self.compRho(0, 0, 0)
        #     # for o in range(1,self.Norb):
        #     #     rho0 = rho0 + self.compRho(o, o, 0)
        #     phi_ortho = self.P_eps(utils.Fzero)
        #     #Compute the orthonormalisation constraint Sum_{j≠i} F^0_ij|phi^1_{j}>
        #     for j in range(self.Norb): 
        #         phi_ortho = phi_ortho + self.Fock[0][orb, j]*self.phi_prev[j][-2][1] 
        #     #Compute the term \hat{rho}^0*\hat{F}^1|phi^0_i>
        #     rhoFphi = self.P_eps(utils.Fzero)
        #     for j in range(self.Norb):
        #         rhoFphi += self.Fock[order][j, orb] * self.phi_prev[j][-2][0]
        #     return -2*self.G_mu[orb]((self.Vnuc[0] + self.J[0])*self.phi_prev[orb][-2][1] - self.K[orb][1] - phi_ortho + Fphi - rhoFphi) 
        # else: #order == 0

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
        if phi_orth == None:
            phi_orth = self.phi_prev
        S = np.zeros((self.Norb, self.Norb)) #Overlap matrix S_i,j = <Phi^i|Phi^j>
        for i in range(self.Norb):
            for j in range(i, self.Norb):
                S[i,j] = vp.dot(phi_orth[i][-1], phi_orth[j][-1]) #compute the overlap of the current ([-1]) step
                if i != j:
                    S[j,i] = np.conjugate(S[i,j])
        return S

    def orthonormalise(self, phi_in = None): #Lödwin orthogonalisation and normalisation
        if phi_in == None:
            phi_in = self.phi_prev
        S = self.computeOverlap(phi_in)
        #Diagonalise S to compute S' := S^-1/2 
        eigvals, U = np.linalg.eigh(S) #U is the basis change matrix

        s = np.diag(np.power(eigvals, -0.5)) #diagonalised S 
        #Compute s^-1/2
        Sprime = np.dot(U,np.dot(s,np.transpose(U))) # S^-1/2 = U^dagger s^-1/2 U

        #Apply S' to each orbital to obtain a new orthogonal element
        phi_ortho = []
        for i in range(self.Norb):
            phi_tmp =  self.P_eps(utils.Fzero)
            for j in range(self.Norb):
            # for j in range(limit[i]):
                phi_tmp = phi_tmp + Sprime[i,j]*phi_in[j][-1]
            phi_tmp.normalize()
            phi_ortho.append(phi_tmp)
        return phi_ortho


class scfsolv_1stpert(scfsolv):
    Fock1 : np.ndarray                       #1st order perturbed Fock matrix
    Vnuc1 : vp.FunctionTree                  #1st order perturbed nuclear potential
    J1 : vp.FunctionTree                     #1st order perturbed Coulomb potential
    K1 : list                                #list of 1st order perturbed exchange potential applied to each orbital
    phi_prev1 : list                         #list of all 1st order perturbed orbitals and their KAIN history
    f_prev1 : list                           #list of all 1st order perturbed orbitals updates and their KAIN history
    E1_n : list                              #list of perturbed orbital energies

    def __init__(self, prec, khist, lgdrOrder=6, sizeScale=-4, nboxes=2, scling=1.0) -> None: 
        super().__init__(prec, khist, lgdrOrder, sizeScale, nboxes, scling)
        self.J1 = super().P_eps(utils.Fzero)
        self.K1 = []
        self.phi_prev1 = []
        self.f_prev1 = []

    def __init__(self, Parent) -> None:
        self.super = Parent
        self.J1 = super().P_eps(utils.Fzero)
        self.K1 = []
        self.phi_prev1 = []
        self.f_prev1 = []

    def init_molec(self) -> None:
        self.Vnuc1 = super().P_eps(lambda r : super().f_nuc(r)) #TODO: CHANGER ÇA SINON C'EST JUSTE scfsolv AVEC DES PAS SUPPLÉMENTAIRES
        #initial guesses can be zero for the perturbed orbitals
        self.phi_prev1 = []
        for i in range(self.Norb):
            self.phi_prev.append([super().P_eps(utils.Fzero)])
        self.f_prev = [[] for i in range(self.Norb)] #list of the corrections at previous steps
        #Compute the Fock matrix and potential operators 
        self.compFock()                                                         #TODO ODODODODODODOODO depuis là
        # print("squalalala",self.Fock)
        #Energies of each orbital
        self.E1_n = []
        #internal nuclear energy
        self.E_pp = super().fpp()
        # Prepare Helmholtz operators
        super().G_mu = []
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
            phi_np1.normalize()
            self.phi_prev1[orb].append(phi_np1)
        #Orthonormalise orbitals since they are molecular orbitals
        phi_ortho = super().orthonormalise(self.phi_prev1)
        for orb in range(self.Norb): #Mandatory loop due to questionable data format choice.
            self.phi_prev1[orb][-1] = phi_ortho[orb]
    
    def expandSolution(self):
        #Orthonormalise orbitals in case they aren't yet
        phi_ortho = super().orthonormalise(self.phi_prev1)
        for orb in range(self.Norb): #Mandatory loop due to questionable data format choice.
            self.phi_prev1[orb][-1] = phi_ortho[orb]
        #Compute the fock matrix of the system
        self.compFock()
        
        phistory = []
        self.E1_n = []
        norm = []
        update = []
        for orb in range(super().Norb):
            self.E1_n.append(self.Fock1[orb, orb])
            #Redefine the Helmholtz operator with the updated energy
            # mu = np.sqrt(-2*super().E_n[orb])
            # self.G_mu[orb] = vp.HelmholtzOperator(self.mra, mu, self.prec) 
            #Compute new power iteration for the Helmholtz operator
            phi_np1 = self.powerIter(orb)        
            #create an alternate history of orbitals which include the power iteration
            phistory.append([phi_np1])
        #Orthonormalise the alternate orbital history
        phistory = super().orthonormalise(phistory)
        # phi_prev = orthonormalise(phi_prev)
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
            #Normalize
            norm.append(phi_n.norm())
            phi_n.normalize()
            #Save new orbital
            self.phi_prev1[orb].append(phi_n)
            #Correction norm (convergence metric)
            update.append(delta.norm())
            if len(self.phi_prev1[orb]) > super().khist: #deleting oldest element to save memory
                del self.phi_prev1[orb][0]
                del self.f_prev1[orb][0]
        return np.array(self.E1_n), np.array(norm), np.array(update)

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

    #computation of operators
    def compFock(self): 
        self.Fock1 = np.zeros((super().Norb, super().Norb))
        self.K1 = []
        self.J1 = self.computeCoulombPot()
        for j in range(super().Norb):
            #Compute the potential operator
            self.K1.append(self.computeExchangePotential(j))
            # V = Vnuc
            # compute the energy from the orbitals 
            Fphi = self.compFop(j)
            for i in range(super().Norb):
                self.Fock1[i,j] = vp.dot(super().phi_prev[i][-1], Fphi)

    def compFop(self, orb): #Computes the Fock operator applied to an orbital orb #TODO: modifier ça pour que ça rentre dans compFock
        Tphi = -0.5*(super().D(super().D(super().phi_prev[orb][-1], 0), 0) + super().D(super().D(super().phi_prev[orb][-1], 1), 1) + super().D(super().D(super().phi_prev[orb][-1], 2), 2)) #Laplacian of the orbitals
        # Fphi = Tphi + self.Vnuc[orderF]*self.phi_prev[orb][-1] + self.J[orderF]*self.phi_prev[orb][-1] - self.K[orb][orderF]
        Fphi = Tphi + self.Vnuc1*super().phi_prev[orb][-1] + self.J1*super().phi_prev[orb][-1] - self.K1[orb]
        return Fphi
    
    def compRho(self, orb1, orb2 ):
        rho = super().phi_prev[orb1][-1]*self.phi_prev1[orb2][-1] + self.phi_prev1[orb1][-1]*super().phi_prev[orb2][-1]
        return rho

    def computeCoulombPot(self): 
        PNbr = 4*np.pi*self.compRho(0,0)
        for orb in range(1, super().Norb):
            PNbr = PNbr + 4*np.pi*self.compRho(orb,orb)
        return self.Pois(2*PNbr) #factor of 2 because we sum over the number of orbitals, not electrons

    def computeExchangePotential(self, idx):
        K = super().phi_prev[0][-1]*self.Pois(4*np.pi*self.compRho(0,idx))
        for j in range(1, super().Norb):
            K = K + super().phi_prev[j][-1]*self.Pois(4*np.pi*self.compRho(j,idx))
        return K 

    def powerIter(self, orb): #TODO: C'est à moitié cassé, à corriger maintenant 
        # if order == 1:
        # print("TODODODODODODODODO")
        Fphi = self.compFop(orb) 
        # rho0 = self.compRho(0, 0, 0)
        # for o in range(1,self.Norb):
        #     rho0 = rho0 + self.compRho(o, o, 0)
        phi_ortho = super().P_eps(utils.Fzero)
        #Compute the orthonormalisation constraint Sum_{j≠i} F^0_ij|phi^1_{j}>
        for j in range(super().Norb): 
            # phi_ortho = phi_ortho + self.Fock[0][orb, j]*self.phi_prev[j][-2][1] 
            phi_ortho = phi_ortho + super().Fock[orb, j]*self.phi_prev1[j][-1] 
        #Compute the term \hat{rho}^0*\hat{F}^1|phi^0_i>
        rhoFphi = super().P_eps(utils.Fzero)
        for j in range(super().Norb):
            rhoFphi += self.Fock1[j, orb] * super().phi_prev[j][-2][0]
        return -2*super().G_mu[orb]((super().Vnuc[0] + super().J[0])*self.phi_prev1[orb][-1] - super().K[orb] - phi_ortho + Fphi - rhoFphi) 
 