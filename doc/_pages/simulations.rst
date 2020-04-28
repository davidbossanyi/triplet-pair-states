Simulations
===========
	
.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
Introduction
------------

The :class:`tripletpairs.kineticmodelling.KineticSimulation` class provides a convenient wrapper for performing simulations, incorporating both time-resolved and steady-state models, full spin Hamiltonian, magnetic field effects and averaging over things like euler angles to mimic amorphous morphologies.
   
Example
-------

.. code-block:: python

   from tripletpairs.kineticmodelling import timeresolvedmodels
   from tripletpairs.kineticmodelling import KineticSimulation
   import numpy as np
   # set up the kinetic model
   m = timeresolvedmodels.Merrifield()
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
   # set up the magnetic parameters
   J = 0
   D = 5e-6
   E = D/3
   X = D/1000
   rAB = (np.cos(86.37*np.pi/180), 0, np.sin(86.37*np.pi/180))
   alpha = np.pi/2
   beta = -118.32*np.pi/180
   gamma = np.pi/2
   theta = np.linspace(0, np.pi, 21)
   phi = 0
   B = np.linspace(0, 0.25, 10)
   # do the simulation
   sim = KineticSimulation(m)
   sim.set_spin_hamiltonian_parameters(J, X, rAB, D, E, alpha, beta, gamma, B, theta, phi)
   sim.simulate_state_populations(['S1', 'TT_total', 'T1'])
   mfe1 = sim.calculate_mfe('S1', time_range=(20, 30))
   mfe2 = sim.calculate_mfe('S1', time_range=(100, 200))

API
---

.. autoclass:: tripletpairs.kineticmodelling.KineticSimulation
	:members: