from tripletpairs.kineticmodelling import steadystatemodels
from tripletpairs.kineticmodelling import KineticSimulation

import numpy as np
from matplotlib import pyplot as plt


###############################################################################
# SET UP THE KINETIC MODEL
###############################################################################

m = steadystatemodels.Merrifield()
m.kSF = 0.05
m.k_SF = 0.05
m.kDISS = 5e-3
m.kTTA = 1e-23
m.kTTNR = 1e-5
m.kTNR = 1e-5
m.kSNR = 0.06
m.G = 1e13


###############################################################################
# SET THE SPIN HAMILTONIAN PARAMETERS
###############################################################################

J = 0
D = 5e-6
E = D/3
X = D/1000
rAB = (np.cos(86.37*np.pi/180), 0, np.sin(86.37*np.pi/180))
alpha = np.pi/2
beta = -118.32*np.pi/180
gamma = np.pi/2
theta = np.pi/4
phi = 0
B = np.linspace(0, 0.25, 10)


###############################################################################
# DO THE SIMULATION
###############################################################################

sim = KineticSimulation(m)
sim.set_spin_hamiltonian_parameters(J, X, rAB, D, E, alpha, beta, gamma, B, theta, phi)
sim.simulate_state_populations(['S1'])
mfe = sim.calculate_mfe('S1')


###############################################################################
# PLOT SOME RESULTS
###############################################################################

fig, ax = plt.subplots(figsize=(6, 4))
# magnetic field effect
ax.plot(1000*B, 100*mfe, color='seagreen')
ax.axhline(0, color='0.5', linewidth=1)
ax.set_xlim([-5, 250])
ax.set_xlabel('Magnetic Field Strength (mT)', fontsize=14)
ax.set_ylabel('MFE (%)', fontsize=14)
ax.tick_params(axis='both', labelsize=14)
