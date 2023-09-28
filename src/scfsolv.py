import matplotlib.pyplot as plt
import numpy as np
# import KAIN
import utils
from vampyr import vampyr3d as vp

# from copy import deepcopy


class scfsolv_wrong:
    world : vp.BoundingBox
    mra : vp.MultiResolutionAnalysis
    prec : float
    P_eps : vp.ScalingProjector             #projection operator
    Pois : vp.PoissonOperator               #Poisson operator
    D : vp.ABGVDerivative                   #derivative operator
    G_mu : list                             #list of Helmholtz operator for every orbital
    Norb : int                              #number of orbitals
    Fock : list                             #Fock matrix
    Vnuc : list                             # list of unperturbed and perturbed nuclear potential
    J : list                                #Coulomb potential
    K : list                                #list of exchange potential applied to each orbital
    phi_prev : list                         #list of all orbitals and their KAIN history
    f_prev : list                           #list of all orbitals updates and their KAIN history
    khist : int                             #KAIN history size 
    R : list                                #list of all coordinates of each atom
    Z : list                                #list of all atomic numbers of each atom
    Nz : int                                #number of atoms
    E_pp : float                            #internal nuclear energy
    Po : int                                #Maximum perturbation order 
    #rho : list                              #Electron density (list of unperturbed and perturbed densities)


    def __init__(self, prec, khist, pert_order = 0, lgdrOrder=6, sizeScale=-4, nboxes=2, scling=1.0) -> None:
        self.world = vp.BoundingBox(corner=[-1]*3, nboxes=[nboxes]*3, scaling=[scling]*3, scale= sizeScale)
        self.mra = vp.MultiResolutionAnalysis(order=lgdrOrder, box=self.world)
        self.prec = prec
        self.P_eps = vp.ScalingProjector(self.mra, self.prec) 
        self.Pois = vp.PoissonOperator(self.mra, self.prec)
        #Derivative operator
        self.D = vp.ABGVDerivative(self.mra, a=0.5, b=0.5)
        #Physical properties (placeholder values)
        self.Norb = 1
        self.Fock = [np.zeros((self.Norb, self.Norb)) for i in range(pert_order+1)]
        # print("Construct Fock",len(self.Fock)) #debug
        self.J = [self.P_eps(utils.Fzero) for i in range(pert_order+1)]
        self.Vnuc = [self.P_eps(utils.Fzero) for i in range(pert_order+1)]
        self.K = []
        self.R = []
        self.Z = []
        self.Nz = 1
        self.E_pp = 0.0
        #Accelerator
        self.khist = khist
        self.phi_prev = []
        self.f_prev = []
        #linear response
        self.Po = pert_order
        # self.rho = [[0 for j in range(self.Po)] for i in range(self.Norb)]

    
    def init_molec(self, No,  pos, Z, init_g_dir) -> None: #unperturbed initialisation
        self.Nz = len(Z)
        self.Norb = No
        self.R = pos
        self.Z = Z
        self.Vnuc[0] = self.P_eps(lambda r : self.f_nuc(r))
        #initial guesses provided by mrchem
        self.phi_prev = []
        for i in range(self.Norb):
            ftree = vp.FunctionTree(self.mra)
            ftree.loadTree(f"{init_g_dir}phi_p_scf_idx_{i}_re") 
            self.phi_prev.append([[ftree]])
        self.f_prev = [[] for i in range(self.Norb)] #list of the corrections at previous steps
        #Compute the unperturbed Fock matrix and potential operators 
        self.compFock() #by default: unperturbed
        #Energies of each orbital
        E_n = []
        #internal nuclear energy
        self.E_pp = self.fpp()
        # Prepare Helmholtz operators
        self.G_mu = []
        # J = computeCoulombPot(phi_prev)
        for i in range(self.Norb):
            E_n.append(self.Fock[0][i,i])
            # E_n.append(2*F[i]+E_pp)
            mu = np.sqrt(-2*E_n[i])
            # mu = 1 #E = -0.5 analytical solution
            self.G_mu.append(vp.HelmholtzOperator(self.mra, mu, self.prec))  # Initalize the operator
        offset = 0 #offset in indices due to questionable orbital storage format, more info in the CompFock method
        for o in range(self.Po+1):
            if o > 0:
                offset = -1
                print("initmolec",len(self.phi_prev[0]))
                self.Vnuc[o] = [self.Vnuc[0]*1e-3]
                for orb in range(self.Norb):
                    self.phi_prev[orb][0].append(self.P_eps(utils.Fzero))
                self.compFock(o)
            for orb in range(self.Norb):
                #First establish a history (at least one step) of corrections to the orbital with standard iterations with Helmholtz operator to create
                # Apply Helmholtz operator to obtain phi_np1 #5
                phi_np1 = self.powerIter(orb, o)
                # Compute update = ||phi^{n+1} - phi^{n}|| #6
                self.f_prev[orb].append([phi_np1 - self.phi_prev[orb][-1+offset][o]])
                phi_np1.normalize()
                if o > 0:
                    print("POUEUTEUTUETUEUTEUTUE")
                    self.phi_prev[orb][-1].append(phi_np1)
                else:
                    self.phi_prev[orb].append([phi_np1])
            #Orthonormalise orbitals since they are molecular orbitals
            phi_ortho = self.orthonormalise(o)
            for orb in range(self.Norb): #Mandatory loop due to questionable data format choice.
                self.phi_prev[orb][-1][o] = phi_ortho[orb]
            
    
    #computation of operators
    def compFock(self, order = 0) -> None: #Computes the Fock matrix
        self.Fock[order] = np.zeros((self.Norb, self.Norb))
        if order == 0:
            self.K = []
        self.J[order] = self.computeCoulombPot(order)
        for j in range(self.Norb):
            #Compute the potential operator
            if order == 0:
                self.K.append([self.computeExchangePotential(j, order)])
            else:
                self.K[j].append(self.computeExchangePotential(j, order))
            # compute the energy from the orbitals 
            Fphi = self.compFop(j, order, 0)
            for i in range(self.Norb):
                print("Comp Fock",len(self.phi_prev), i)
                self.Fock[order][i,j] = vp.dot(self.phi_prev[i][-1][0], Fphi)

    def compFop(self, orb, orderF = 0, orderPhi = 0): #Computes the Fock operator applied to an orbital orb #TODO: modifier ça pour que ça rentre dans compFock
        Tphi = -0.5*(self.D(self.D(self.phi_prev[orb][-1][orderPhi], 0), 0) + self.D(self.D(self.phi_prev[orb][-1][orderPhi], 1), 1) + self.D(self.D(self.phi_prev[orb][-1][orderPhi], 2), 2)) #Laplacian of the orbitals
        print(len(self.K[orb]), orderF, len(self.K))
        # Fphi = Tphi + self.Vnuc[orderF]*self.phi_prev[orb][-1][orderPhi] + self.J[orderF]*self.phi_prev[orb][-1][orderPhi] - self.K[orb][orderF]
        Fphi = Tphi + self.Vnuc[0]*self.phi_prev[orb][-1][orderPhi] + self.J[orderF]*self.phi_prev[orb][-1][orderPhi] - self.K[orb][orderF]
        return Fphi

    def compRho(self, orb1, orb2, order = 0): #todo mettre un offset pour les pertubration
        offset = 0 #because of the way orbitals are stored, once we want to compute perturbed orbitals, the last element in the "kain history" list only contains unperturbed orbitals, we thus have to look one orbital before
        if order > 0:
            offset = -1
        out = self.P_eps(utils.Fzero)
        # print("compRho", len(self.phi_prev[orb1][-1]))
        for i in range(order + 1):
            out = out + self.phi_prev[orb1][-1+offset][i]*self.phi_prev[orb2][-1+offset][order-i]
        return out
            

    def computeCoulombPot(self, order = 0): 
        PNbr = 4*np.pi*self.compRho(0,0, order) #padbug
        for orb in range(1, self.Norb):
            PNbr = PNbr + 4*np.pi*self.compRho(orb,orb, order)#padbug
        return self.Pois(2*PNbr) #factor of 2 because we sum over the number of orbitals, not electrons

    def computeExchangePotential(self, idx, order = 0, orderPhi = 0):
        K = self.phi_prev[0][-1][orderPhi]*self.Pois(4*np.pi*self.compRho(0, idx, order))#padbug
        for j in range(1, self.Norb):
            K = K + self.phi_prev[j][-1][orderPhi]*self.Pois(4*np.pi*self.compRho(j, idx, order))#padbug
        return K 
    
    def expandSolution(self, order=0): #TODO: modifier 1er ordre
        #Orthonormalise orbitals in case they aren't yet
        print("orthonorm phi_prev ", len(self.phi_prev[0][-1]), len(self.phi_prev[1][-1]), len(self.phi_prev[2][-1]),len(self.phi_prev[3][-1]),len(self.phi_prev[4][-1]))
        offset = 0
        if order>0:
            offset = -1
        phi_ortho = self.orthonormalise(order, offset=offset) #TODO: mettre l'offset d'indices ici pour order>0
        for orb in range(self.Norb): #Mandatory loop due to questionable data format choice.
            print(len(self.phi_prev[0][-1]))
            print("orb =", orb, order)
            self.phi_prev[orb][-1][order] = phi_ortho[orb]
        #Compute the fock matrix of the system
        self.compFock(order)
        phistory = []
        E_n = []
        norm = []
        update = []
        for orb in range(self.Norb):
            E_n.append(self.Fock[order][orb, orb])
            #Redefine the Helmholtz operator with the updated energy
            mu = np.sqrt(-2*E_n[orb])
            self.G_mu[orb] = vp.HelmholtzOperator(self.mra, mu, self.prec) 
            #Compute new power iteration for the Helmholtz operatort 
            phi_np1 = self.powerIter(orb, order)
            #create an alternate history of orbitals which include the power iteration
            phistory.append([[phi_np1]])
        #Orthonormalise the alternate orbital history
        print("orthonorm Expanddong")
        phistory = self.orthonormalise(0, phistory) #here order=0 because phistory will only contain the power iteration for the current choice of perturbation order (i.e. `order` is implicit and doesn't need to be specified a second time, else out-of-range indices are what you seek)
        # phi_prev = orthonormalise(phi_prev)
        for orb in range(self.Norb): 
            self.f_prev[orb].append([phistory[orb] - self.phi_prev[orb][-1][order]])
            #Setup and solve the linear system Ac=b
            c = self.setuplinearsystem(orb, order)
            #Compute the correction delta to the orbitals 
            delta = self.f_prev[orb][-1][order]
            for j in range(len(self.phi_prev[orb])-1):
                delta = delta + c[j]*(self.phi_prev[orb][j][order] - self.phi_prev[orb][-1][order] + self.f_prev[orb][j][order] - self.f_prev[orb][-1][order])
            #Apply correction
            phi_n = self.phi_prev[orb][-1][order]
            phi_n = phi_n + delta
            #Normalize
            norm.append(phi_n.norm())
            phi_n.normalize()
            #Save new orbital
            if order > 0:
                print("TOODODODODODOD")
            else:
                self.phi_prev[orb].append([phi_n])
            #Correction norm (convergence metric)
            update.append(delta.norm())
            if len(self.phi_prev[orb]) > self.khist: #deleting oldest element to save memory
                del self.phi_prev[orb][0]
                del self.f_prev[orb][0]
        return np.array(E_n), np.array(norm), np.array(update)
    
    def powerIter(self, orb, order=0):
        if order == 1:
            # print("TODODODODODODODODO")
            Fphi = self.compFop(orb, 1, 0)
            # rho0 = self.compRho(0, 0, 0)
            # for o in range(1,self.Norb):
            #     rho0 = rho0 + self.compRho(o, o, 0)
            phi_ortho = self.P_eps(utils.Fzero)
            #Compute the orthonormalisation constraint Sum_{j≠i} F^0_ij|phi^1_{j}>
            for j in range(self.Norb): 
                phi_ortho = phi_ortho + self.Fock[0][orb, j]*self.phi_prev[j][-2][1] 
            #Compute the term \hat{rho}^0*\hat{F}^1|phi^0_i>
            rhoFphi = self.P_eps(utils.Fzero)
            for j in range(self.Norb):
                rhoFphi += self.Fock[order][j, orb] * self.phi_prev[j][-2][0]
            return -2*self.G_mu[orb]((self.Vnuc[0] + self.J[0])*self.phi_prev[orb][-2][1] - self.K[orb][1] - phi_ortho + Fphi - rhoFphi) 
        else: #order == 0
            #Compute phi_ortho := Sum_{j!=i} F_ij*phi_j 
            phi_ortho = self.P_eps(utils.Fzero) 
            for orb2 in range(self.Norb):
                #Compute off-diagonal Fock matrix elements
                if orb2 != orb:
                    phi_ortho = phi_ortho + self.Fock[order][orb, orb2]*self.phi_prev[orb2][-1][order]
            return -2*self.G_mu[orb]((self.Vnuc[order] + self.J[0])*self.phi_prev[orb][-1][0] - self.K[orb][0] - phi_ortho)

    def setuplinearsystem(self,orb, order=0):
        lenHistory = len(self.phi_prev[orb])
        # Compute matrix A
        A = np.zeros((lenHistory-1, lenHistory-1))
        b = np.zeros(lenHistory-1)
        for l in range(lenHistory-1):  
            dPhi = self.phi_prev[orb][l][order] - self.phi_prev[orb][-1][order]
            b[l] = vp.dot(dPhi, self.f_prev[orb][-1][order])
            for j in range(lenHistory-1):
                A[l,j] = -vp.dot(dPhi, self.f_prev[orb][j][order] - self.f_prev[orb][-1][order])
        #solve Ac = b for c
        c = np.linalg.solve(A, b)
        return c

    def scfRun(self, thrs = 1e-3,  printVal = False, pltShow = False):
        update = np.ones(self.Norb)
        norm = np.zeros(self.Norb)
        #iteration counter
        i = 0
        # Optimization loop (KAIN) 
        while update.max() > thrs:
            print(f"=============Iteration: {i}")
            i += 1 
            E_n, norm, update = self.expandSolution(0)
            # for o in range(1,self.Po+1):
            #     self.expandSolution(o)
            for orb in range(self.Norb):
                # this will plot the wavefunction at each iteration
                r_x = np.linspace(-5., 5., 1000)
                phi_n_plt = [self.phi_prev[orb][-1][0]([x, 0.0, 0.0]) for x in r_x]
                plt.plot(r_x, phi_n_plt) 
                
                if printVal:
                    print(f"Orbital: {orb}    Norm: {norm}    Update: {update}    Energy:{E_n}")
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

    def computeOverlap(self, order = 0, phi_orth = None, offset = 0):
        # offset = 0
        # if order>0:
        #     offset = -1
        # pert_order = 0
        if phi_orth == None:
            phi_orth = self.phi_prev
            # pert_order = order
        S = np.zeros((self.Norb, self.Norb)) #Overlap matrix S_i,j = <Phi^i|Phi^j>
        for i in range(self.Norb):
            for j in range(i, self.Norb):
                S[i,j] = vp.dot(phi_orth[i][-1+offset][order], phi_orth[j][-1+offset][order]) #compute the overlap of the current ([-1]) step #TODO: le problème est list index out of range dans cette ligne
                if i != j:
                    S[j,i] = np.conjugate(S[i,j])
        return S


    def orthonormalise(self, order=0, phi_in = None, offset = 0): #Lödwin orthogonalisation and normalisation
        # offset = 0
        # if order>0:
        #     offset = -1
        if phi_in == None:
            phi_in = self.phi_prev
        S = self.computeOverlap(order, phi_in, offset=0)
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
                phi_tmp = phi_tmp + Sprime[i,j]*phi_in[j][-1+offset][order]
            phi_tmp.normalize()
            phi_ortho.append(phi_tmp)
        return phi_ortho
