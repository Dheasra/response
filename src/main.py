from vampyr import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt
# from copy import deepcopy

# import KAIN
# import utils
from scfsolv import scfsolv as ss

prec = 1.0e-4
khist = 5

r_x = np.linspace(-16, 16, 10000) # create an evenly spaced set of points between -0.99 and 0.99
r_y = np.linspace(-16, 16, 10000)
r_z = np.linspace(-16, 16, 10000)
r = [r_x, r_y, r_z]


# Analytic nuclear potential
# print("pouet")
#LiH
# R, Z, pathMol, nOrb = [[1.508001,  0.188973, 0.188973], [-1.508001,  0.188973, 0.188973]], [3,1], "/home/qpitto/Tests_KAIN/test_LiH/norun/orbitals/phi_p_scf_", 2 #LiH
#Ne
# R, Z, pathMol, nOrb = [[0.100, 0.100, 0.100]], [10], "/home/qpitto/Tests_KAIN/test_Ne/norun/orbitals/phi_p_scf_", 5 #Ne
#H2O
# R, Z, pathMol, nOrb = [[0.0000,  0.000, -0.125], [-1.4375,  0.000,  1.025], [1.4375,  0.000,  1.025]], [8,1,1], "/home/qpitto/Tests_KAIN/initial_guesses/phi_p_scf_", 5 #H2O
#CH4
R, Z, pathMol, nOrb = [[0.1, 0.1, 0.1], [0.1, 0.1, 2.15413215], [2.03631019, 0.1, -0.5856834], [-0.54635362, 1.92524633, -0.5856834], [-0.54635362, -1.72524633, -0.5856834]], [6,1,1,1,1], "/home/qpitto/Tests_KAIN/test_CH4/norun/orbitals/phi_p_scf_", 5 #CH4
nz = len(Z)

#create scf solver instance with default MRA
Scfs = ss(prec, khist)
#initialise molecule geometry and properties
Scfs.init_molec(nOrb, R, Z, pathMol)


# thrs = 1.0e-3
update = np.ones(nOrb)
norm = np.zeros(nOrb)
# phi_prev_U = phi_prev

#iteration counter
i = 0

# Optimization loop (KAIN) #TODO continuer
while update.max() > thrs:
    print(f"=============Iteration: {i}")
    i += 1 

    E_n, norm, update = Scfs.expandSolution()
    # phi_prev, f_prev, E_n, norm, update = expandSolution(phi_prev, f_prev)

    for orb in range(nOrb):
        # phi_prev = orthonormalise(phi_prev)

        # print("Babawiw" ,vp.dot(phi_prev[0][-1],phi_prev[1][-1]))

    
        # this will plot the wavefunction at each iteration
        r_x = np.linspace(-5., 5., 1000)
        phi_n_plt = [phi_prev[orb][-1]([x, 0.0, 0.0]) for x in r_x]
        plt.plot(r_x, phi_n_plt) 
        
        print(f"Orbital: {orb}    Norm: {norm}    Update: {update}    Energy:{E_n}")
        # printDebug(phi_prev, orb, orb, Vnuc, J, K)
    # phi_prev = orthonormalise(phi_prev)
    
    # phi_prev, f_prev = applyFock(phi_prev, f_prev)

# plt.legend([f"iteration:{i} " for i in range(5)])

# plt.plot(r_x, [f_phi([x, 0., 0.]) for x in r_x], '.-', color='pink')

plt.show()

