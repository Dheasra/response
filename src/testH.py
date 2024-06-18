from vampyr import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt

from scfsolv import scfsolv

#modified remrchem
def gs_D_1e(spinorb1, potential, mra, prec, thr, derivative):
    print('One-electron calculations')
    
    error_norm = 1
    #compute_last_energy = False

    light_speed = spinorb1.light_speed
    old_energy = 0
    delta_e = 1
    while (error_norm > thr and delta_e > thr/1000):
        # hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
        # v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
        hd_psi = 
        add_psi = hd_psi + v_psi
        energy = spinorb1.dot(add_psi).real
        mu = orb.calc_dirac_mu(energy, light_speed)
        tmp = orb.apply_helmholtz(v_psi, mu, prec)
#        tmp = orb.apply_dirac_hamiltonian(v_psi, prec, energy, der = derivative)
        tmp.cropLargeSmall(prec)
        new_orbital = orb.apply_dirac_hamiltonian(tmp, prec, energy, der = derivative)
#        new_orbital =  orb.apply_helmholtz(tmp, mu, prec)
#        new_orbital.cropLargeSmall(prec)
        new_orbital.cropLargeSmall(prec)
        new_orbital.normalize()
        delta_psi = new_orbital - spinorb1
        #orbital_error = delta_psi.dot(delta_psi).real
        deltasq = delta_psi.squaredNorm()
        error_norm = np.sqrt(deltasq)
        print('Error', error_norm)
        delta_e = np.abs(energy - old_energy)
        print('Delta E', delta_e)
        print('Energy',energy - light_speed**2)
        old_energy = energy
        spinorb1 = new_orbital
    
    hd_psi = orb.apply_dirac_hamiltonian(spinorb1, prec, der = derivative)
    v_psi = orb.apply_potential(-1.0, potential, spinorb1, prec)
    add_psi = hd_psi + v_psi
    energy = spinorb1.dot(add_psi).real
    energy_1s = analytic_1s(light_speed, 1, -1, 1)
    print("Exact energy: ", energy_1s - light_speed**2)
    print('Final Energy:',energy - light_speed**2)
    print('Delta Energy:',energy - old_energy)
    print('Error Energy:',energy - energy_1s)
    return spinorb1