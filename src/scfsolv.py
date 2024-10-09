from vampyr import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy


#Stuff to import complex_fcn from far away
import sys
import os
# Construct the absolute path to the directory containing the module.
# module_path = os.path.abspath("/home/qpitto/Tests_KAIN/ZORA/ReMRChem2C/orbital4c")
# # Append the module path to sys.path
# sys.path.append(module_path)

# import KAIN
import utils
from complex_fcn import complex_fcn, apply_poisson
# from complex_tree import complex_tree
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
    Nspinor : int                           #number of orbitals
    Fock : np.ndarray                       #Fock matrix
    Vnuc : spinor                           #nuclear potential
    J : spinor                              #Coulomb potential
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
    Kinetic_op : list                       #Kinetic operator applied to each orbitals. Used for debugging
    Nel : int                               #Number of electrons
    scalar : bool                           #Selection between scalar ZORA or spin-orbit interations


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
    def __init__(self, Ncomponents, prec, khist, lgdrOrder=6, sizeScale=-4, nboxes=2, scling=1.0, scalar = True, lightspeed = 137.03599) -> None:
        # print("init")
        self.world = vp.BoundingBox(corner=[-1]*3, nboxes=[nboxes]*3, scaling=[scling]*3, scale= sizeScale)
        self.mra = vp.MultiResolutionAnalysis(order=lgdrOrder, box=self.world)
        self.prec = prec
        self.P_eps = vp.ScalingProjector(self.mra, self.prec) 
        self.Pois = vp.PoissonOperator(self.mra, self.prec)
        #Derivative operator
        # self.D = vp.ABGVDerivative(self.mra, a=0.5, b=0.5)
        self.D = vp.BSDerivative(self.mra, 1) #bspline derivative
        #Physical properties (placeholder values)
        self.Nspinor = 0
        self.Fock = np.zeros((self.Nspinor, self.Nspinor), dtype=complex)
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
        self.c = lightspeed #in atomic units
        self.Ncomp = Ncomponents
        # testZORA = complex_fcn(self.mra)
        # self.Kinetic = spinor(self.mra, Ncomponents)
        # self.Kinetic.setZero() 
        self.Kinetic_op = []
        self.scalar = scalar

        self.tutiter = 0

    # This method initialises all physical properties of the system.
    # No[in]: integer giving the number of orbitals. Be warned that this code currently only works for closed shell systems.
    # pos[in]: List of lists (Matrix-ish) of floating point number. The mother list contains lists (vectors) of the 3D coordinates of the each atom in the system, in atomic units.
    # Z[in]: List of integers. Each number is the atomic charge of an atom in the system. Be warned that each atoms must be ordered in the same way in pos and Z
    # source_init_guess[in]: Directory (string) of the inital guesses for the orbitals. Currently this code cannot create an initial guess, and thus needs an MRChem-compatible inital guess created from another program, such as MRChem.
    def initMolec(self, No,  pos, Z, source_init_guess, restricted = True, sclr = True, copy_orbitals = False) -> None: 
        print("Initmolec")
        self.Nz = len(Z)
        self.R = pos
        self.Z = Z
        self.Nspinor = No
        self.Nel = 0
        self.scalar = sclr
        # computing the number of electrons (Nel)
        for i in range(self.Nz):
            self.Nel += Z[i]
        # Kinetic = spinor(self.mra, self.Ncomp)
        # Kinetic.setZero()
        for i in range(len(self.Vnuc)):
            print("Vnuc init", i)
            self.Vnuc.compVect[i].real = self.P_eps(lambda r : self.f_nuc(r, threshold=-1e5))
        # for j in range(self.Ncomp):
        #     print("is Vnuc constant???", j, utils.is_constant(self.Vnuc.compVect[j]))
        # #initial guesses provided by mrchem
        for i in range(self.Nspinor):
            # print("init guess", i)
            phi = spinor(self.mra, self.Ncomp)
            phi.setZero()
            # paired_idx = int(np.floor(i/2))
            paired_idx = i
            print("paired idx", paired_idx)
            if copy_orbitals:
                phi = source_init_guess.phi_prev[i][-1].reproject(self.P_eps)
            elif restricted == True: #Paired orbitals
                # print("tut")
                phi.compVect[0].real.loadTree(f"{source_init_guess}phi_p_scf_idx_{paired_idx}_re")  
                # phi.compVect[1].real.loadTree(f"{source_init_guess}phi_p_scf_idx_{paired_idx}_re")  #test to see if the initial guess was an issue
                # phi.normalize() #test to see if the initial guess was an issue
            else:
                if i%2 == 0:
                    # print(f"{source_init_guess}phi_a_scf_idx_{paired_idx}_re")
                    phi.compVect[0].real.loadTree(f"{source_init_guess}phi_a_scf_idx_{paired_idx}_re")
                else: 
                    phi.compVect[0].real.loadTree(f"{source_init_guess}phi_b_scf_idx_{paired_idx}_re")
            # print("init guess norm", vp.dot(phi.compVect[0].real, phi.compVect[0].real))
            self.phi_prev.append([phi])
        self.f_prev = [[] for i in range(self.Nspinor)] #list of the corrections at previous steps

        self.printOperators()
        # print("Test Pauli matrix multiplication")
        # #TODO
        # for i in range(self.Nspinor):
        #     # print("init guess", i)
        #     phi = spinor(self.mra, self.Ncomp)
        #     phi.setZero()
        #     phi.compVect[0].real.loadTree(f"{init_g_dir}phi_p_scf_idx_{i}_re") 
        #     for direction in range(3):
        #         phi_dir = utils.apply_Pauli(direction, phi)
        #         print("Test direction: ", direction)
        #         print("Overlap Phi / simga*Phi" ,phi.dot(phi_dir))
        #         print("Norm Phi:", phi.compNorm(), "Norm sigma*Phi", phi_dir.compNorm())

        # print("Overlap init")
        # print("S=",self.computeOverlap())
        # TestImag = 1j*self.phi_prev[0][-1]
        # print("InitMolec test imag: ", complex_fcn.dot(self.phi_prev[0][-1],TestImag))
        #Compute the Fock matrix and potential operators 
        # print("Fock start")
        print("pouet")
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
        for i in range(self.Nspinor):
            # print("Helmholtz operator init ", i)
            self.E_n.append(self.Fock[i,i])
            # print("Energy: ", self.Fock[i, i], i)
            mu = np.sqrt(-2*self.E_n[i])
            # print("mu", mu)
            self.G_mu.append(vp.HelmholtzOperator(self.mra, mu, self.prec))  # Initalize the operator
        for orb in range(self.Nspinor):
            # print("powerIter", orb)
            #First establish a history (at least one step) of corrections to the orbital with standard iterations with Helmholtz operator to create
            # Apply Helmholtz operator to obtain phi_np1 #5
            phi_np1 = self.powerIter(orb)
            # print("post powerIter", orb)
            # Compute update = ||phi^{n+1} - phi^{n}|| #6
            # print("Post power Iter, pre normalize", orb, complex_fcn.dot(phi_np1, phi_np1))
            self.f_prev[orb].append(phi_np1 - self.phi_prev[orb][-1])
            phi_np1.normalize()
            # print("Post power Iter, post normalize", orb, complex_fcn.dot(phi_np1, phi_np1))
            self.phi_prev[orb].append(phi_np1)

            if len(self.phi_prev[orb]) > self.khist: #deleting the first element if we don't want to use KAIN
                del self.phi_prev[orb][0]
                # del self.f_prev[orb][0]
        # print("Overlap init post-powerIter")
        # print("S=",self.computeOverlap())
        #Orthonormalise orbitals since they are molecular orbitals
        phi_ortho = self.orthonormalise()
        # phi_ortho = self.orthonormaliseKramer()
        for orb in range(self.Nspinor): #Mandatory loop due to questionable data format choice.
            self.phi_prev[orb][-1] = phi_ortho[orb]
        # print("Overlap init end")
            print(orb, " length update history: ", len(self.f_prev[orb]))
        print("Init molec S=",self.computeOverlap())

        

    #===Computation of operators===
    
    # This method computes the Fock matrix of the system
    def compFock(self): 
        # print("compFock")
        self.Fock = np.zeros((self.Nspinor, self.Nspinor), dtype=complex)
        self.K = []
        self.J = self.computeCoulombOperator()
        #===Original, non-Hermitian Fock matrix (even though it should be)
        for j in range(self.Nspinor):
            print("compFock main loop", j)
            #Compute the potential operator
            self.K.append(self.computeExchangePotential(j))            
            Fphi = self.compFop(j)
            # compute the energy from the orbitals 
            for i in range(self.Nspinor):
                # self.Fock[i,j] = complex_fcn.dot(self.phi_prev[i][-1], Fphi) #TODO 2C
                # print("compFock val", self.phi_prev[i][-1].dot(self.Vnuc * self.phi_prev[j][-1]))
                self.Fock[i,j] = self.phi_prev[i][-1].dot(Fphi) 
        # #===Test new, forced-Hermitian Fock matrix 
        # for j in range(self.Nspinor):
        #     # print("compFock main loop", j)
        #     #Compute the potential operator
        #     self.K.append(self.computeExchangePotential(j))            
        #     Fphi = self.compFop(j)
        #     # compute the energy from the orbitals 
        #     for i in range(j, self.Nspinor):
        #         # self.Fock[i,j] = complex_fcn.dot(self.phi_prev[i][-1], Fphi) #TODO 2C
        #         # print("compFock val", self.phi_prev[i][-1].dot(self.Vnuc * self.phi_prev[j][-1]))
        #         self.Fock[i,j] = self.phi_prev[i][-1].dot(Fphi) 
        #         if i != j:
        #             self.Fock[j,i] = np.conjugate(self.Fock[i,j])

    # This method computes the Fock operator applied to an orbital "orb"; returns F_\phi = \hat{F}\ket{\phi} 
    # orb[in]: integer index of the chosen orbital
    # Fphi[out]: function tree (vp.FunctionTree) representation of the operator applied to \ket{\phi[idx]}
    def compFop(self, orb): 
        # print("start compFop")
        #Zora potential
        V_z = self.Vnuc + self.J
        # print("WARNING: V_Z only contains  Vnuc")
        # V_z = self.Vnuc 
        # print("V_z", type(V_z), V_z.dot(V_z), len(V_z))
        # for j in range(self.Ncomp):
            # print("is Vz constant???", j, utils.is_constant(V_z.compVect[j]))
        #     print("Vnuc check", self.Vnuc.compVect[i].dot(self.Vnuc.compVect[i]))
        #constant fct f(x) = 1
        # one = spinor(self.mra, self.Ncomp) #TODO: make a map to compute 1-Vz 
        # one.setZero()
        # for j in range(self.Ncomp):
        #     one.compVect[j].real = self.P_eps(lambda r : utils.Fone(r))
        #     # print("is one constant???", j, utils.is_constant(one.compVect[j]))
        # print("compFop one done")
        #kappa operator
        kappa = spinor(self.mra, self.Ncomp)
        kappa.setZero()
        Gmap = vp.FunctionMap(fmap = self.computeKappaMinus1, prec=self.prec)
        # print("kappa before")
        for j in range(self.Ncomp): #TODO: use function mapping
            print("kappa compo", j)
            # kappa.compVect[j].real = self.P_eps(lambda r: self.computeKappaMinus1(r,j))
            # V_z = self.Vnuc + self.J
            # vp.advanced.map(lambda r : prec=self.prec, out=kappa, inp=V_z, fmap = self.computeKappaMinus1(r, j))
            kappa.compVect[j].real = Gmap(V_z.compVect[j].real) 
            # kappa.compVect[j].real = self.P_eps(lambda r: np.real(V_z(j,r))*((4*self.c**2 - 2*np.real(V_z(j,r)))**(-1))) #representing kappa -1/2, w/ kappa = 0.5 *(1-V/2c^2)^-1
            # kappa.compVect[j].real = self.P_eps(lambda r: 0.5*(1-np.real(V_z(j, r))/(2*self.c**2))**(-1)) #TODO: Erreur ici parce que je fais ^-1 à un objet qui n'a pas de valeur imaginaire (=0) donc ça fait 1/0 et ça retourne NaN 
        print("kappa after")
        kappa.crop(self.prec)
        # self.plotTree(kappa.compVect[0].real)
        # for j in range(self.Ncomp):
        #     print("is kappa constant???", j, utils.is_constant(kappa.compVect[j]))
        # kappa_m1 = one-V_z/(2*self.c**2) 
        # print("kappa^-1 squared norm", kappa_m1.compSqNorm())
        # print("================kappa", kappa.dot(kappa), kappa.compSqNorm(), "================")
        dkappa = [kappa.derivative(0).crop(self.prec), kappa.derivative(1).crop(self.prec), kappa.derivative(2).crop(self.prec)]
        # print("---------derivative kappa =", dkappa[0].compVect[0].real, dkappa[1].compVect[0].real, dkappa[2].compVect[0].real)
        # for i in range(len(dkappa)):
        #     dkappa.crop(self.prec)
            # print("compo: ", i)
            # print("dkappa", dkappa[i].dot(dkappa[i]))
        print("================kappadone==================")
        dphi = [self.phi_prev[orb][-1].derivative(0).crop(self.prec), self.phi_prev[orb][-1].derivative(1).crop(self.prec), self.phi_prev[orb][-1].derivative(2).crop(self.prec)] 
        print("compFop kappa done")
        # for i in range(len(dphi)):
        #     print("compo: ", i)
        #     for j in range(len(dphi[i])):
        #         print("dphi", type(dphi[i].compVect[j]))
            # print("dphi", dphi[i].dot(dphi[i]))
            # print("dkappa", dkappa[i].dot(dkappa[i]))
        #Kinetic operator computation 
        #first scalar term: (kappa-1/2)*nabla^2
        kNab2 = spinor(self.mra, self.Ncomp)
        kNab2.setZero()
        kNab2 = kappa * dphi[0].derivative(0)
        kNab2 = kNab2 + kappa * dphi[1].derivative(1)
        kNab2 = kNab2 + kappa * dphi[2].derivative(2)
        kNab2 =  kNab2.crop(self.prec)
        #second part of the first term, to correct for the use of kappa-1/2, so -1/2 * nabla^2
        halfNab2 = spinor(self.mra, self.Ncomp)
        halfNab2.setZero()
        halfNab2 = 0.5*dphi[0].derivative(0)
        halfNab2 = halfNab2 + 0.5*dphi[1].derivative(1)
        halfNab2 = halfNab2 + 0.5*dphi[2].derivative(2)
        halfNab2 =  halfNab2.crop(self.prec)


        # print("compFop kNab2 done")
        #second scalar term: -0.5 *nabla(kappa) * nabla 
        NabkNab = spinor(self.mra, self.Ncomp)
        NabkNab.setZero()
        NabkNab = dkappa[0] * dphi[0]
        # NabkNab = NabkNab.crop(self.prec)
        NabkNab = NabkNab + dkappa[1] * dphi[1]
        # NabkNab = NabkNab.crop(self.prec)
        NabkNab = NabkNab + dkappa[2] * dphi[2]
        NabkNab = NabkNab.crop(self.prec)
        print("compFop NabkNab done") #"where kappa^-1 here?", no kappa^-1 here because it's not the scf eqt here, it's the hamiltonian

        #--spin orbit term-- #TODO: test to see if there is an issue with memory SCALARDEBUG
        #x direction
        sporb = spinor(self.mra, self.Ncomp)
        sporb.setZero()
        # print("sporb init ok")
        sporb = 1j * utils.apply_Pauli(0,dkappa[1]*dphi[2]-dkappa[2]*dphi[1]).crop(self.prec)
        # pouet = utils.apply_Pauli(0,(dkappa[1]*dphi[2]-dkappa[2]*dphi[1]))
        # print("sporb 1 ", sporb.dot(sporb), pouet.dot(pouet))
        # print("sporb step 1 ok")
        sporb = sporb  + 1j * utils.apply_Pauli(1,dkappa[2]*dphi[0]-dkappa[0]*dphi[2]).crop(self.prec)
        # pouet = utils.apply_Pauli(1,(dkappa[2]*dphi[0]-dkappa[0]*dphi[2]))
        # print("sporb 2 ", sporb.dot(sporb), pouet.dot(pouet))
        # print("sporb step 2 ok")
        sporb = sporb + 1j * utils.apply_Pauli(2,dkappa[0]*dphi[1]-dkappa[1]*dphi[0]).crop(self.prec)
        # pouet = utils.apply_Pauli(2,-0.5*(dkappa[0]*dphi[1]-dkappa[1]*dphi[0]))
        # print("sporb 3 ", sporb.dot(sporb), pouet.dot(pouet))
        # print("sporb step 3 ok")
        sporb = sporb.crop(self.prec)
        print("compFop sporb done")

        #Total kinetic operator
        Tphi = spinor(self.mra, self.Ncomp)
        Tphi = (-1)*(kNab2 + halfNab2 + NabkNab) 
        # Tphi = (-1)*(kNab2 + NabkNab) 
        if self.scalar == False:
            Tphi = Tphi - sporb #TODO: test to see if there is an issue with memory SCALARDEBUG
        self.Kinetic_op.append(Tphi)
        # print("types CompFop", type(self.Vnuc), type(self.J), type(self.phi_prev[orb][-1]), self.K[orb])
        Fphi = spinor(self.mra, self.Ncomp)
        Fphi = Tphi + self.Vnuc*self.phi_prev[orb][-1] + self.J*self.phi_prev[orb][-1] - self.K[orb]
        # print("WARNING: J and K are not included in Fock")
        # Fphi = Tphi + self.Vnuc*self.phi_prev[orb][-1] 
        # print("Test compFop", Tphi.dot(Tphi), (self.Vnuc * self.phi_prev[orb][-1]).dot(self.Vnuc * self.phi_prev[orb][-1]), (self.J * self.phi_prev[orb][-1]).dot(self.J * self.phi_prev[orb][-1]), self.K[orb].dot(self.K[orb]))
        # print("Test compFop", Tphi.dot(Tphi), self.phi_prev[orb][-1].dot(self.Vnuc * self.phi_prev[orb][-1]), self.phi_prev[orb][-1].dot(self.J * self.phi_prev[orb][-1]), self.phi_prev[orb][-1].dot(self.K[orb]))
        # print("CompFop", orb, complex_fcn.dot(self.J * self.phi_prev[orb][-1], self.J * self.phi_prev[orb][-1]))
        # Fphi_test = Tphi
        print("CompFop real", orb, Fphi.dot(Fphi))
        return Fphi

    #Computes the product between an electron in orbital orb1 and another in orbital orb2
    # orb1[in]: integer index of the left (bra) element of the product 
    # orb2[in]: integer index of the right (ket) element of the product
    # rho[out]: function tree (vp.FunctionTree) representation of the product
    def compProduct(self, orb1, orb2): #TODO: 
        # # print("compProduct") 
        # #NOTE: to compute the density, a factor 2 will be lacking, as we are counting over full orbitals, not electrons
        # # but in this code it is added where needed
        # rho = self.phi_prev[orb1][-1]*self.phi_prev[orb2][-1]
        # return rho
        if orb1 >= self.Nspinor and orb2 < self.Nspinor:
            rho = self.phi_prev[orb2][-1].multKramer(self.phi_prev[orb1-self.Nspinor][-1])
            rho = rho.conjugate()
            # # this will plot the wavefunction at each iteration
            # r_x = np.linspace(-5., 5., 1000
            # comp_alpha = [rho(0, [x, 0.0, 0.0]) for x in r_x]
            # plt.plot(r_x, comp_alpha) 
            # comp_beta = [rho(1, [x, 0.0, 0.0]) for x in r_x]
            # plt.plot(r_x, comp_beta) 
            print("orb1 Is rho zero? (should be)", utils.is_constant(rho.compVect[0], self.prec, 0.0), utils.is_constant(rho.compVect[1], self.prec, 0.0))
        elif orb2 >= self.Nspinor and orb1 < self.Nspinor:
            rho = self.phi_prev[orb1][-1].multKramer(self.phi_prev[orb2-self.Nspinor][-1])
            print("orb2 Is rho zero? (should be)", utils.is_constant(rho.compVect[0], self.prec, 0.0), utils.is_constant(rho.compVect[1], self.prec, 0.0))
        else:
            rho = self.phi_prev[orb1%self.Nspinor][-1].conjugate()*self.phi_prev[orb2%self.Nspinor][-1]
        return rho

    #This method computes the Coulomb operator  of the molecule
    # [out]: function tree (vp.FunctionTree) representation of the operator
    def computeCoulombOperator(self): 
        # sum_e = 0
        # for i in range(len(self.Z)):
        #     sum_e += self.Z[i]
        if self.Nel > 1:
            # print("compute Coulomb more that 1 e")
            # print("comp J")
            # output = spinor(self.mra, self.Ncomp)
            PNbr = 4*np.pi*self.compProduct(0, 0) #Factor 2 because the density is given by the sum of the "normal" spinors and their kramer-paired spinors, and the latter is exactly the same as the first one
            for orb in range(1, 2*self.Nspinor):
                PNbr = PNbr + 4*np.pi*self.compProduct(orb, orb)
            # return self.Pois(2*PNbr) #factor of 2 because we sum over the number of orbitals, not electrons
            # output.real = self.Pois(2*PNbr.real)
            # output.imag = self.Pois(2*PNbr.imag)
            return utils.apply_Poisson_spinor(self.Pois, PNbr)
            # return output
        else: 
            print("compute Coulomb onlz 1 e")
            J = spinor(self.mra, self.Ncomp)
            J.setZero()
            return J
    

    #This method computes the exchange operator applied to an orbital of index "idx"
    # idx[in]: integer index of the chosen orbital in the list of orbitals "phi_prev"
    # [out]: function tree (vp.FunctionTree) representation of the operator
    def computeExchangePotential(self, idx):
        # sum_e = 0
        # for i in range(len(self.Z)):
        #     sum_e += self.Z[i]
        if self.Nel > 1:
            # print("comp K")
            # K = complex_fcn(self.mra)
            # Pois_term = complex_fcn(self.mra)
            # Pois_term.real = self.Pois(4*np.pi*self.compProduct(0, idx).real)
            # Pois_term.imag = self.Pois(4*np.pi*self.compProduct(0, idx).imag)
            # K.real = (self.phi_prev[0][-1]*Pois_term).real
            # K.imag = (self.phi_prev[0][-1]*Pois_term).imag
            # for j in range(1, self.Nspinor):
            #     Pois_term.real = self.Pois(4*np.pi*self.compProduct(j, idx).real)
            #     Pois_term.imag = self.Pois(4*np.pi*self.compProduct(j, idx).imag)
            #     K.real = K.real + (self.phi_prev[j][-1]*Pois_term).real
            #     K.imag = K.imag + (self.phi_prev[j][-1]*Pois_term).imag
            #     # K = K + self.phi_prev[j][-1]*self.Pois(4*np.pi*self.compProduct(j, idx))
            # return K 
            Korb = spinor(self.mra, self.Ncomp)
            Korb.setZero()
            for j in range(2*self.Nspinor): #TODO: ça risque de faire des double-comptages ça 
                Korb = Korb + self.phi_prev[j%self.Nspinor][-1] * utils.apply_Poisson_spinor(self.Pois, 4*np.pi*self.compProduct(j, idx))
            return Korb
            # Korb = self.phi_prev[0][-1] * complex_fcn.apply_poisson(4 * np.pi * self.phi_prev[0][-1].density(self.prec), self.mra, self.Pois, self.prec)
            # for j in range(1, self.Nspinor):
            #     Korb += self.phi_prev[j][-1] * complex_fcn.apply_poisson(4 * np.pi * self.phi_prev[j][-1].density(self.prec), self.mra, self.Pois, self.prec)
            # return Korb
        else: 
            print("compute Korb only 1 e")
            Korb = spinor(self.mra, self.Ncomp)
            Korb.setZero()
            return Korb
    

    #This is the main method of the class; it moves a step forward in the optimisation of the orbitals
    #It makes use of a KAIN accelerator to improves convergence speed.
    #The updated orbitals and history are stored in the class' attributes, and deleted when too old in the history.
    #E_n[out]: Energy level of each orbital in atomic units, outputed in a numpy array
    #norm[out]: norm of each orbital, given in a numpy array
    #update[out]: norm of the difference between the newest and second newest orbital in the history, given in a numpy array.
    def expandSolution(self):
        # print("expand sol")
        #Orthonormalise orbitals in case they aren't yet
        phi_ortho = self.orthonormalise()
        # phi_ortho = self.orthonormaliseKramer()
        for orb in range(self.Nspinor): #Mandatory loop due to questionable data format choice.
            self.phi_prev[orb][-1] = phi_ortho[orb]
        #Compute the fock matrix of the system
        self.compFock()

        print("ExpandSol start S=", self.computeOverlap())
        
        phistory = []
        self.E_n = []
        norm = []
        update = []
        shell_energy, U = np.linalg.eig(self.Fock)
        print("Shell energy", shell_energy)
        for orb in range(self.Nspinor):
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
        # phistory = self.orthonormaliseKramer(phistory)
        # phi_prev = orthonormalise(phi_prev)

        #Orthogonalisation test
        phi_test = []
        for o in range(self.Nspinor):
            phi_test.append([phistory[o]])
        print("ExpandSol S=", self.computeOverlap(phi_test))
        
        for orb in range(self.Nspinor):
            self.f_prev[orb].append(phistory[orb] - self.phi_prev[orb][-1])
            #Setup and solve the linear system Ac=b
            c = self.setupLinearSystem(orb)
            #Compute the correction delta to the orbitals 
            delta = self.f_prev[orb][-1]
            for j in range(len(self.phi_prev[orb])-1):
                delta = delta + c[j]*(self.phi_prev[orb][j] - self.phi_prev[orb][-1] + self.f_prev[orb][j] - self.f_prev[orb][-1])
            delta.crop(self.prec)
            #Apply correction
            phi_n = self.phi_prev[orb][-1]
            phi_n = phi_n + delta
            #Normalize
            # norm_Re  = phi_n.real.norm()
            # norm_Im  = phi_n.imag.norm()
            # norm.append(np.sqrt(norm_Re**2 + norm_Im**2))
            norm.append(np.sqrt(phi_n.compSqNorm()))
            phi_n.normalize()
            #Save new orbital
            self.phi_prev[orb].append(phi_n)
            #Correction norm (convergence metric)
            update.append(np.sqrt(delta.compSqNorm())) 
            if len(self.phi_prev[orb]) > self.khist: #deleting oldest element to save memory
                del self.phi_prev[orb][0]
                del self.f_prev[orb][0]
        print("ExpandSol final S=", self.computeOverlap())
        self.printOperators()
        print("end ExpandSol")
        return np.array(self.E_n), np.array(norm), np.array(update)
    
    def expandSolution_nokain(self):
        print("expand sol no kain")
        #Orthonormalise orbitals in case they aren't yet
        phi_ortho = self.orthonormalise()
        for orb in range(self.Nspinor): #Mandatory loop due to questionable data format choice.
            self.phi_prev[orb][-1] = phi_ortho[orb]
        #Compute the fock matrix of the system
        self.compFock()
        
        phistory = []
        self.E_n = []
        norm = []
        update = []
        for orb in range(self.Nspinor):
            self.E_n.append(self.Fock[orb, orb])
            #Redefine the Helmholtz operator with the updated energy
            mu = np.sqrt(-2*self.E_n[orb])
            self.G_mu[orb] = vp.HelmholtzOperator(self.mra, mu, self.prec) 
            #Compute new power iteration for the Helmholtz operator
            phi_np1 = self.powerIter(orb)        
        #     #create an alternate history of orbitals which include the power iteration
        #     phistory.append([phi_np1])
        # #Orthonormalise the alternate orbital history
        # phistory = self.orthonormalise(phistory)
        # # phi_prev = orthonormalise(phi_prev)
        # for orb in range(self.Nspinor):
        #     self.f_prev[orb].append(phistory[orb] - self.phi_prev[orb][-1])
        #     #Setup and solve the linear system Ac=b
        #     c = self.setupLinearSystem(orb)
        #     #Compute the correction delta to the orbitals 
        #     delta = self.f_prev[orb][-1]
        #     for j in range(len(self.phi_prev[orb])-1):
        #         delta = delta + c[j]*(self.phi_prev[orb][j] - self.phi_prev[orb][-1] + self.f_prev[orb][j] - self.f_prev[orb][-1])
        #     delta.crop(self.prec)
            #Apply correction
            print("length",len(self.f_prev), len(self.f_prev[orb]),)
            self.f_prev[orb][-1] = phi_np1 - self.phi_prev[orb][-1]
            # self.phi_prev[orb][-1] = phi_np1
            #Normalize
            # norm_Re  = phi_n.real.norm()
            # norm_Im  = phi_n.imag.norm()
            # norm.append(np.sqrt(norm_Re**2 + norm_Im**2))
            norm.append(np.sqrt(phi_np1.compSqNorm()))
            phi_np1.normalize()
            #Save new orbital
            self.phi_prev[orb][-1] = phi_np1
            #Correction norm (convergence metric)
            update.append(np.sqrt(self.f_prev[orb][-1].compSqNorm())) 
            if len(self.phi_prev[orb]) > 1: #no kain means there shouldn't be any kain history
                print("pouet")
                del self.phi_prev[orb][0]
                del self.f_prev[orb][0]
        return np.array(self.E_n), np.array(norm), np.array(update)
    
    #This method executes one SCF "power iteration", i.e. one application of the Helmholtz operator to the SCF equation
    #orb[in]: integer index of the chosen orbital in the list of orbitals "phi_prev"
    #[out]: function tree (vp.FunctionTree) representation of the updated orbital with index "orb"
    def powerIter(self, orb):
        # print("power Iter")
         #Zora potential
        V_z = self.Vnuc + self.J
        # V_z = self.Vnuc
        # #constant fct f(x) = 1
        # one = spinor(self.mra, self.Ncomp)
        # one.setZero()
        # for j in range(self.Ncomp):
        #     one.compVect[j].real = self.P_eps(utils.Fone)
        # one.crop(self.prec)
        #kappa operator
        # kappa_m1 = one-V_z/(2*self.c**2)
        kappa = spinor(self.mra, self.Ncomp) #represents kappa - 1/2 for simplicity
        kappa_m1 = spinor(self.mra, self.Ncomp) #represents the inverse of kappa (NOT kappa -1/2)
        # kappa_minus_1 = self.P_eps(lambda r: np.real(V_z(0, r))*((4*self.c**2 - 2*np.real(V_z(0,r)))**(-1))).crop(self.prec)
        Gmap = vp.FunctionMap(fmap = self.computeKappaMinus1, prec=self.prec)
        # kappa_alt = spinor(self.mra, self.Ncomp)
        for j in range(self.Ncomp):
            # print("kappa compo", j)
            # kappa_alt.compVect[j].real = self.P_eps(lambda r: 0.5*(1-np.real(V_z(j, r))/(2*self.c**2))**(-1)) #TODO: checker les 1/2
            kappa.compVect[j].real = Gmap(V_z.compVect[j].real) 
            kappa_m1.compVect[j].real = self.P_eps(lambda r: 1-np.real(V_z(j, r))/(2*self.c**2)) 
            # kappa.compVect[j].real = kappa_minus_1
            # kappa.compVect[j].real = self.P_eps(lambda r: self.computeKappaMinus1(r,j))
            # kappa_m1.compVect[j].real = self.P_eps(lambda r: np.real(4*self.c**2/np.real(V_z(j,r)) - 2))
        kappa_m1 = kappa_m1.crop(self.prec)
        # print("Kappa = ",kappa.compVect[0].real)
        # print("Kappa_alt = ",kappa_alt.compVect[0].real)
        # kappa = kappa_m1**(-1)
        kappa = kappa.crop(self.prec)
        # print("Power Iter Kappa ok")

        #TODO: check les 1/2 et ajuster les kappa pour qu'ils deviennent des kappa - 1/2 

        dkappa = [kappa.derivative(0).crop(self.prec), kappa.derivative(1).crop(self.prec), kappa.derivative(2).crop(self.prec)]
        dphi = [self.phi_prev[orb][-1].derivative(0).crop(self.prec), self.phi_prev[orb][-1].derivative(1).crop(self.prec), self.phi_prev[orb][-1].derivative(2).crop(self.prec)]

        #First SCF term (potential-like)
        Vkphi = (self.Vnuc + self.J)*kappa_m1*self.phi_prev[orb][-1] - self.K[orb]*kappa_m1
        Vkphi.crop(self.prec)
        VzFphi = V_z/(self.c**2) * self.Fock[orb, orb] * self.phi_prev[orb][-1] 
        VzFphi.crop(self.prec)
        P_scf = (Vkphi + VzFphi).crop(self.prec)
        # print("P I T3 ok")

        # print("P I derivatives ok")
        #Second SCF term (Scalar kinetic term)
        So_scl = spinor(self.mra, self.Ncomp)
        So_scl.setZero()
        for i in range(3): 
            # print("Term 1", i)
            So_scl = So_scl + kappa_m1*dkappa[i]*dphi[i]
        So_scl = So_scl.crop(self.prec)
        # print("P I T1 ok")
        
        #Second SCF term (spin-orbit kinetic term) #TODO: test to see if there is an issue with memory SCALARDEBUG
        So_so = spinor(self.mra, self.Ncomp)
        So_so.setZero()
        So_so = 1j * kappa_m1 * utils.apply_Pauli(0,dkappa[1]*dphi[2]-dkappa[2]*dphi[1]).crop(self.prec)
        So_so = So_so + 1j * kappa_m1 * utils.apply_Pauli(1,dkappa[2]*dphi[0]-dkappa[0]*dphi[2]).crop(self.prec)
        So_so = So_so + 1j * kappa_m1 * utils.apply_Pauli(2,dkappa[0]*dphi[1]-dkappa[1]*dphi[0]).crop(self.prec)
        # So_so = -1j * kappa_m1 # * utils.apply_Pauli(0,-0.5*(dkappa[1]*dphi[2]-dkappa[2]*dphi[1]))
        # So_so = So_so - 1j * kappa_m1 # * utils.apply_Pauli(1,-0.5*(dkappa[2]*dphi[0]-dkappa[0]*dphi[2]))
        # So_so = So_so - 1j * kappa_m1 # * utils.apply_Pauli(2,-0.5*(dkappa[0]*dphi[1]-dkappa[1]*dphi[0]))
        # print("P I T2 ok")
        So_so = So_so.crop(self.prec)
        
        #Fourth SCF term (Non-canonical basis correction)
        Fij_phij = spinor(self.mra, self.Ncomp)
        Fij_phij.setZero()
        for orb2 in range(self.Nspinor):
            # print("PowerIter debug: ", type(self.Fock[orb, orb2]*self.phi_prev[orb2][-1]), self.Fock[orb, orb2]*self.phi_prev[orb2][-1],  "tut", type(self.phi_prev[orb2][-1]))
            #Compute off-diagonal Fock matrix elements
            if orb2 != orb:
                # Fij_phij = Fij_phij + self.Fock[orb, orb2]*self.phi_prev[orb2][-1]
                Fij_phij = Fij_phij + self.Fock[orb, orb2]*self.phi_prev[orb2][-1]
        Fij_phij = Fij_phij*kappa_m1
        Fij_phij.crop(self.prec)
        # print("P I T4 ok")

        phi_np1 = spinor(self.mra, self.Ncomp)
        phi_np1_tmp = spinor(self.mra, self.Ncomp)
        # phi_np1_tmp = So_scl + So_so + P_scf + Fij_phij #oldSCALARDEBUG
        phi_np1_tmp = P_scf - So_scl - Fij_phij
        if self.scalar == False:
            phi_np1_tmp = phi_np1_tmp - So_so 
        for l in range(self.Ncomp):
            if(phi_np1_tmp.compVect[l].real.squaredNorm() > 1e-12):
                # print("PowerIter Orb ok", orb)
                # print("PowerIter, pre Helmholtz, SCF equation", orb, complex_fcn.dot(phi_np1_tmp, phi_np1_tmp))
                phi_np1.compVect[l].real = -2 * self.G_mu[orb](phi_np1_tmp.compVect[l].real)
                # print("PowerIter, post Helmholtz", orb, complex_fcn.dot(phi_np1, phi_np1))
            if(phi_np1_tmp.compVect[l].imag.squaredNorm() > 1e-12):
                phi_np1.compVect[l].imag = -2 * self.G_mu[orb](phi_np1_tmp.compVect[l].imag)
        phi_np1.crop(self.prec)
        return phi_np1
    
    #This method sets up then solve the linear system Ac=b for a specific orbital of idex "orb"
    #This operation can be done independantly for each orbital as the ground state orbitals are separated.
    #For the following, the index "orb" will be ommited for the documentation.
    #The matrix elements are defined as A_ij = <phi_history[i]-phi_history[newest]|update_history[j]-update_history[newest]>
    #The vector elements are defined as B_i = <phi_history[i]-phi_history[newest]|update_history[newest]>
    #orb[in]: integer index of the chosen orbital in the list of orbitals "phi_prev"
    #c[out]: numpy array, solution of the linear system
    def setupLinearSystem(self,orb):
        # print("setup Lin Syst")
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
        print("scf run")
        update = np.ones(self.Nspinor)
        norm = np.zeros(self.Nspinor)

        #iteration counter
        i = 0

        # Optimization loop (KAIN) #TODO continuer
        while update.max() > thrs:
            print(f"=============Iteration: {i}")
            i += 1 
            print("F=", self.Fock)
            print("S=",self.computeOverlap())
            if self.khist > 0:
                self.E_n, norm, update = self.expandSolution()
            else:
                self.E_n, norm, update = self.expandSolution_nokain()

            self.printOperators()
            for orb in range(self.Nspinor):
                # # this will plot the wavefunction at each iteration
                # r_x = np.linspace(-5., 5., 1000)
                # phi_n_plt = [self.phi_prev[orb][-1]([x, 0.0, 0.0]) for x in r_x]
                # plt.plot(r_x, phi_n_plt) 
                if pltShow:
                    plt.show()
                
                if printVal:
                    print(f"Orbital: {orb}    Norm: {norm}    Update: {update}    Energy:{self.E_n}")
                    if self.Nspinor == 1:
                        print("Scaled energy (4C Dirac energy)", (1/self.E_n.real - 1/(self.c**2))**(-1))

        # if pltShow:
        #     plt.show()

    #Utilities
    
    #Method to compute the electron-nuclear contribution to the energy.
    #r[in]: position in 3D space; 3-element vector-like. 
    #[out]: Value of the el-nuc potential at the position "r" in space
    def f_nuc(self, r, threshold = -1e5):   
    # def f_nuc(self, r):   
        # print("fnuc")
        out = 0
        #electron-nucleus interaction
        for i in range(self.Nz): 
            val = -self.Z[i]/np.sqrt((r[0]-self.R[i][0])**2 + (r[1]-self.R[i][1])**2 + (r[2]-self.R[i][2])**2)
            if val < threshold:
                val = threshold
            out += val
            # out += -self.Z[i]/np.sqrt((r[0]-self.R[i][0])**2 + (r[1]-self.R[i][1])**2 + (r[2]-self.R[i][2])**2) 
        return out
    
    # def computeKappaMinus1(self, r, component, threshold=1e-2):
    #     close_idx = -1
    #     # print("computekappa start")
    #     for nuc in range(len(self.Z)):
    #         dist = np.sqrt((r[0]-self.R[nuc][0])**2 + (r[1]-self.R[nuc][1])**2 + (r[2]-self.R[nuc][2])**2)
    #         if dist < threshold:
    #             close_idx = nuc
    #     if close_idx != -1: 
    #         Vz_eff = self.Z[close_idx] 
    #         close_dist = np.sqrt((r[0]-self.R[close_idx][0])**2 + (r[1]-self.R[close_idx][1])**2 + (r[2]-self.R[close_idx][2])**2)
    #         for nuc in range(len(self.Z)):
    #             if nuc != close_idx:
    #                 dist = np.sqrt((r[0]-self.R[nuc][0])**2 + (r[1]-self.R[nuc][1])**2 + (r[2]-self.R[nuc][2])**2)
    #                 Vz_eff += self.Z[nuc] * close_dist/dist
    #         Vz_eff += self.J(component, r)*close_dist  #Only coulomb, no exchange because exchange doesn't appear in the ZORA effective potential in kappa
    #         return Vz_eff/(2*self.c**2 - Vz_eff)
    #     else:
    #         V_z = self.Vnuc + self.J
    #         return np.real(V_z(component, r))/(4*self.c**2 - 2*np.real(V_z(component,r)))
    #     # # === test ===
    #     # self.tutiter += 1
    #     # print("compKappa-1", self.tutiter)
    #     # V_z = self.Vnuc + self.J
    #     # return np.real(V_z(component, r))/(4*self.c**2 - 2*np.real(V_z(component,r)))

    def computeKappaMinus1(self, VzComponent):
        return np.real(VzComponent)/(4*self.c**2 - 2*np.real(VzComponent))

    
    #Method to compute the nuclear-nuclear contribution to the energy.
    #[out]: Value of the nuclear-nuclear potential at the position "r" in space
    def fpp(self):
        # print("fpp")
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
        # print("compOverlap")
        if phi_orth == None:
            phi_orth = self.phi_prev
            length = self.Nspinor
        else: 
            length = len(phi_orth)
        S = np.zeros((length, length), dtype=complex) #Overlap matrix S_i,j = <Phi^i|Phi^j>
        for i in range(length):
            for j in range(i, length):
                S[i,j] = phi_orth[i][-1].dot(phi_orth[j][-1]) #compute the overlap of the current ([-1]) step
                # S[i,j] = phi_orth[j][-1].dot(phi_orth[i][-1]) #compute the overlap of the current ([-1]) step
                if i != j:
                    S[j,i] = np.conjugate(S[i,j])
        # print("S", S)
        return S
    
    #Method that computes overlap of Kramer's pairs between the orbitals
    #spin_type[in]: Alpha or Beta-spin electrons. 0 corresponds to alpha and 1 to beta
    def computeKramersOverlap(self, spin_type = 0, phi_orth = None): 
        # print("compOverlap")
        if phi_orth == None:
            phi_orth = self.phi_prev
            length = self.Nspinor
        else: 
            length = len(phi_orth)

        S = np.zeros((length, length), dtype=complex) #Overlap matrix S_i,j = <Phi^i|Phi^j>
        for i in range(length):
            for j in range(i, length):
                S[i,j] = phi_orth[i][-1].dotKramer(phi_orth[j][-1]) #compute the overlap of the current ([-1]) step
                # S[i,j] = phi_orth[j][-1].dot(phi_orth[i][-1]) #compute the overlap of the current ([-1]) step
                if i != j:
                    S[j,i] = np.conjugate(S[i,j])
        # print("S", S)
        return S


    #Lödwin orthogonalisation and normalisation
    #phi_orth[in]: List of List of function trees. The reason for this peculiar choice is the data format of the orbitals and their history. 
    #This input argument can be used to orthonormalise some other object.
    #normalise[in]: Boolean value to toggle on/off the normalisation, if unnecessary.
    def orthonormalise(self, phi_in = None, normalise = True):
        # === Orthogonalise orbitals with each other ===
        print("orthonorm")
        if phi_in == None:
            phi_in = self.phi_prev
            length = self.Nspinor
        else: 
            length = len(phi_in)
        S = self.computeOverlap(phi_in)
        #Diagonalise S to compute S' := S^-1/2 
        eigvals, U = np.linalg.eigh(S) #U is the basis change matrix
        # print("orthonorm eigvals=", eigvals)

        s = np.diag(np.power(eigvals, -0.5)) #diagonalised S 
        #Compute s^-1/2
        Sprime = np.dot(U,np.dot(s,np.transpose(U))) # S^-1/2 = U^dagger s^-1/2 U

        #Apply S' to each orbital to obtain a new orthogonal element
        phi_ortho = []
        phi_test = [] #test 
        for i in range(length):
            # phi_tmp =  self.P_eps(utils.Fzero)
            phi_tmp = spinor(self.mra, self.Ncomp)
            phi_tmp.setZero()
            for j in range(length):
            # for j in range(limit[i]):
                phi_tmp = phi_tmp + Sprime[i,j]*phi_in[j][-1]
                # phi_tmp = phi_tmp + Sprime[j,i]*phi_in[j][-1] #TODO: Test to orthonormalise
            if normalise: 
                phi_tmp.normalize()
            phi_tmp.crop(self.prec)
            phi_ortho.append(phi_tmp)
            phi_test.append([phi_tmp]) #test
        print("Post-Ortho S=", self.computeOverlap(phi_test))#test
        return phi_ortho
    
    def orthonormaliseKramer(self, phi_in = None, normalise = True):
        # === Orthonormalise orbitals according to Karmer's paiers
        if phi_in == None:
            phi_in = self.phi_prev
            length = self.Nspinor
        else: 
            length = len(phi_in)

        Sprime = []
        for i in range(min(2, self.Ncomp)):
            S = self.computeKramersOverlap(phi_in)
            #Diagonalise S to compute S' := S^-1/2 
            eigvals, U = np.linalg.eigh(S) #U is the basis change matrix
            # print("orthonorm eigvals=", eigvals)

            s = np.diag(np.power(eigvals, -0.5)) #diagonalised S 
            #Compute s^-1/2
            # Sprime = np.dot(U,np.dot(s,np.transpose(U))) # S^-1/2 = U^dagger s^-1/2 U
            Sprime.append(np.dot(U,np.dot(s,np.transpose(U)))) # S^-1/2 = U^dagger s^-1/2 U

        #Apply S' to each orbital to obtain a new orthogonal element
        phi_ortho = []
        phi_test = [] #test 
        for i in range(length):
            # phi_tmp =  self.P_eps(utils.Fzero)
            phi_tmp = spinor(self.mra, self.Ncomp)
            phi_tmp.setZero()
            for j in range(length):
            # for j in range(limit[i]):
                for k in range(self.Ncomp):
                    phi_tmp.compVect[k] = phi_tmp.compVect[k] + Sprime[k%2][i,j]*phi_in[j][-1].compVect[k]
                # phi_tmp = phi_tmp + Sprime[j,i]*phi_in[j][-1] #TODO: Test to orthonormalise
            if normalise: 
                phi_tmp.normalize()
            phi_tmp.crop(self.prec)
            phi_ortho.append(phi_tmp)
            phi_test.append([phi_tmp]) #test
        print("Post-Ortho S=", self.computeOverlap(phi_test))#test
        return phi_ortho
    
        #Prints the operators' matrices in the basis of the unperturbed orbitals in the terminal. 
    #The best debugger in existance. 
    def printOperators(self):
        self.compFock()
        kinetic = np.zeros((self.Nspinor,self.Nspinor), dtype=complex)
        Vnuc = np.zeros((self.Nspinor,self.Nspinor), dtype=complex)
        coulomb = np.zeros((self.Nspinor,self.Nspinor), dtype=complex)
        exchange = np.zeros((self.Nspinor,self.Nspinor), dtype=complex)
        kramer = np.zeros((self.Nspinor,self.Nspinor), dtype=complex)
        for orb1 in range(self.Nspinor):

            Jphi0 = self.J*self.phi_prev[orb1][-1]
            Vnucphi0 = self.Vnuc*self.phi_prev[orb1][-1]
            for orb2 in range(self.Nspinor):
                kinetic[orb1,orb2] = self.phi_prev[orb2][-1].dot(self.Kinetic_op[orb1])
                Vnuc[orb1,orb2] = self.phi_prev[orb2][-1].dot(Vnucphi0) 
                coulomb[orb1,orb2] = self.phi_prev[orb2][-1].dot(Jphi0) 
                exchange[orb1,orb2] = self.phi_prev[orb2][-1].dot(self.K[orb1])
                kramer[orb1, orb2] = self.phi_prev[orb2][-1].dotKramer(self.phi_prev[orb1][-1])
                # print("correlation", self.phi_prev[orb2][-1].dot(self.phi_prev[orb1][-1]), self.phi_prev[orb1][-1].dot(self.phi_prev[orb2][-1]))
        print("Fock", self.Fock)
        print("Kinetic", kinetic)
        print("Vnuc", Vnuc)
        print("Coulomb", coulomb)
        print("Exchange", exchange)
        print("Kramer pair overlap", kramer)


    def plotTree(self, Ftree, dir = 0):
        # this will plot the wavefunction at each iteration
        r = np.linspace(-5., 5., 1000)
        if dir == 0:
            phi_n_plt = [Ftree([x, 0.0, 0.0]) for x in r]
        elif dir == 1:
            phi_n_plt = [Ftree([0.0, x, 0.0]) for x in r]
        else:
            phi_n_plt = [Ftree([0.0, 0.0, x]) for x in r]

        plt.plot(r, phi_n_plt) 
        # if pltShow:
        plt.show()



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
        for i in range(self.Nspinor):
            self.phi_prev1.append([self.P_eps(utils.Fzero)]) 
        self.f_prev1 = [[] for i in range(self.Nspinor)] #list of the corrections at previous steps
        self.compFock()
        # self.printOperators() 

        self.E1_n, update = self.expandSolution()
        
        pert_mat = np.zeros((self.Nspinor,self.Nspinor))
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
        for orb in range(self.Nspinor):
            self.E1_n.append(self.Fock1[orb, orb])
            #Compute new power iteration for the Helmholtz operator
            phi_np1 = self.powerIter(orb)     
            #create an alternate history of orbitals which include the power iteration
            phistory.append([phi_np1]) 

        # Orthogonalise the alternate orbital history w.r.t. the unperturbed orbital
        phistory = self.orthogonalise(phistory) 

        for orb in range(self.Nspinor):

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
        for orb in range(self.Nspinor):
            self.E1_n.append(self.Fock1[orb, orb])
            #Compute new power iteration for the Helmholtz operator
            phi_np1 = self.powerIter(orb)     
            #create an alternate history of orbitals which include the power iteration
            phistory.append([phi_np1])
        # Orthogonalise the alternate orbital history w.r.t. the unperturbed orbital
        phistory = self.orthogonalise(phistory)

        for orb in range(self.Nspinor):
            self.f_prev1[orb].append(phistory[orb] - self.phi_prev1[orb][-1])

        c = self.setupLinearSystem_all() 

        for orb in range(self.Nspinor):
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
        for orb in range(self.Nspinor):
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
        update = np.ones(self.Nspinor)
        norm = np.zeros(self.Nspinor)
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
            for orb in range(self.Nspinor):
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
        self.Fock1 = np.zeros((self.Nspinor, self.Nspinor))
        self.K1 = []
        self.J1 = self.computeCoulombOperator()
        for j in range(self.Nspinor):
            #Compute the potential operator
            self.K1.append(self.computeExchangePotential(j))
            # compute the energy from the orbitals 
            Fphi = self.compFop(j)
            for i in range(self.Nspinor):
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
        for orb in range(1, self.Nspinor):
            PNbr = PNbr + 4*np.pi*self.compProduct(orb,orb)
        return self.Pois(2*PNbr) #factor of 2 because we sum over the number of orbitals, not electrons
    
    #This method computes the 1st order perturbed exchange operator applied to an unperturbed orbital of index "idx"
    # idx[in]: integer index of the chosen orbital in the list of orbitals "phi_prev"
    # K_idx[out]: function tree (vp.FunctionTree) representation of the operator*orbital 
    def computeExchangePotential(self, idx):
        K_idx = self.phi_prev[0][-1]*self.Pois(4*np.pi*self.phi_prev1[0][-1]*self.phi_prev[idx][-1]) + self.phi_prev1[0][-1]*self.Pois(4*np.pi*self.phi_prev[0][-1]*self.phi_prev[idx][-1])
        for j in range(1, self.Nspinor): #summing over occupied orbitals
            K_idx = K_idx + self.phi_prev[j][-1]*self.Pois(4*np.pi*self.phi_prev1[j][-1]*self.phi_prev[idx][-1]) + self.phi_prev1[j][-1]*self.Pois(4*np.pi*self.phi_prev[j][-1]*self.phi_prev[idx][-1])
        return K_idx 
    
    #This method computes the UNperturbed exchange operator applied to a 1st order perturbed orbital of index "idx"
    # idx[in]: integer index of the chosen orbital in the list of orbitals "phi_prev1"
    # K_idx[out]: function tree (vp.FunctionTree) representation of the operator*orbital 
    def computeUnperturbedExchangePotential(self, idx):
        K_idx = self.phi_prev[0][-1]*self.Pois(4*np.pi*self.phi_prev[0][-1]*self.phi_prev1[idx][-1])
        for j in range(1, self.Nspinor): #summing over occupied orbitals
            K_idx = K_idx + self.phi_prev[j][-1]*self.Pois(4*np.pi*self.phi_prev[j][-1]*self.phi_prev1[idx][-1])
        return K_idx 

    #This method executes one SCF "power iteration", i.e. one application of the Helmholtz operator to the 1st order perturbed SCF equation
    #orb[in]: integer index of the chosen orbital in the list of orbitals "phi_prev"
    #[out]: function tree (vp.FunctionTree) representation of the updated orbital with index "orb" 
    def powerIter(self, orb): #Devrait suivre la méthode qu'utilise MRChem plus précisément
        phi_ortho = self.P_eps(utils.Fzero)
        #Take into account the non-canonical constraint Sum_{j≠i} F^0_ij|phi^1_{j}>
        for j in range(self.Nspinor): 
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
        S = np.zeros((self.Nspinor, self.Nspinor)) #Overlap matrix S_i,j = <Phi^i|Phi^j>
        for i in range(len(phi_in)):
            for j in range(self.Nspinor):
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
            for j in range(self.Nspinor):
                phi_tmp = phi_tmp - S[i,j]*self.phi_prev[j][-1]
            phi_ortho.append(phi_tmp)
        return phi_ortho

    #Prints the operators' matrices in the basis of the unperturbed orbitals in the terminal. 
    #The best debugger in existance. 
    def printOperators(self):
        self.compFock()
        coulomb = np.zeros((self.Nspinor,self.Nspinor))
        exchange = np.zeros((self.Nspinor,self.Nspinor))
        for orb1 in range(self.Nspinor):
            J1phi0 = self.J1*self.phi_prev[orb1][-1]
            for orb2 in range(self.Nspinor):
                coulomb[orb1,orb2] = vp.dot(self.phi_prev[orb2][-1], J1phi0) 
                exchange[orb1,orb2] = vp.dot(self.phi_prev[orb2][-1], self.K1[orb1])
                print("correlation", vp.dot(self.phi_prev[orb2][-1], self.phi_prev[orb1][-1]), vp.dot(self.phi_prev[orb1][-1], self.phi_prev[orb2][-1]))
        print("Fock", self.Fock1)
        print("Coulomb", coulomb)
        print("Exchange", exchange)
