from tripletpairs.kineticmodelling import models
from tripletpairs.kineticmodelling import KineticSimulation

import numpy as np
from matplotlib import pyplot as plt


###############################################################################
# SET UP THE KINETIC MODEL
###############################################################################

m = models.Merrifield()
m.kGEN = 1.8
m.kSF = 0.05
m.k_SF = 0.05
m.kDISS = 5e-3
m.kTTA = 1e-23
m.kRELAX = 0
m.kSSA = 0
m.kTTNR = 1e-5
m.kTNR = 1e-5
m.kSNR = 0.06
m.G = 1e17


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
sim.simulate_state_populations(['S1', 'TT_total', 'T1'], time_resolved=True)
mfe1 = sim.calculate_mfe('S1', time_range=(20, 30))
mfe2 = sim.calculate_mfe('S1', time_range=(100, 200))


###############################################################################
# PLOT SOME RESULTS
###############################################################################

fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={'hspace': 0.3, 'height_ratios': [2, 1.5]}, figsize=(6, 8))
# kinetics at B = 0
ax = axes[0]
ax.loglog(sim.times, sim.state_populations['S1'][:, 0], color='darkred', label=r'S$_1$')
ax.loglog(sim.times, sim.state_populations['TT_total'][:, 0], color='blue', label='TT')
ax.loglog(sim.times, sim.state_populations['T1'][:, 0], color='purple', label=r'T$_1$')
ax.set_xlim([1, 1e5])
ax.set_ylim([1e12, 1e18])
ax.set_xlabel('Time (ns)', fontsize=14)
ax.set_ylabel(r'Population (cm$^{-3}$)', fontsize=14)
ax.tick_params(axis='both', labelsize=14)
ax.legend(fontsize=14, frameon=False)
# magnetic field effect
ax = axes[1]
ax.plot(1000*B, 100*mfe1, color='seagreen', label='20-30ns')
ax.plot(1000*B, 100*mfe2, color='lime', label='100-200ns')
ax.axhline(0, color='0.5', linewidth=1)
ax.set_xlim([-5, 250])
ax.set_xlabel('Magnetic Field Strength (mT)', fontsize=14)
ax.set_ylabel('MFE (%)', fontsize=14)
ax.tick_params(axis='both', labelsize=14)
ax.legend(fontsize=14, frameon=False)
