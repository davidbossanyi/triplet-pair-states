Models
======

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Introduction
------------

The simulations work by solving kinetic models, i.e. systems of rate equations. In the steady-state, the time derivatives are set equal to zero, allowing an exact algebraic solution. For time-resolved simulations, the rate equations are integrated to obtain numerical excited-state populations as a function of time.
   
Diagrams
--------

Draw some pictures of the different models, showing states and rates!

Example
-------

.. code-block:: python

   m = timeresolvedmodels.Merrifield()
   # set the rate constants
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
   # set the initial excitation density
   m.G = 1e17
   # simulate
   m.simulate()
   # get some results
   S1 = m.simulation_results['S1']

Time-Resolved Models API
------------------------

.. autoclass:: tripletpairs.kineticmodelling.timeresolvedmodels.MerrifieldExplicit1TT
	:inherited-members:
	:members:
	
.. autoclass:: tripletpairs.kineticmodelling.timeresolvedmodels.Merrifield
	:inherited-members:
	:members:
	
.. autoclass:: tripletpairs.kineticmodelling.timeresolvedmodels.Bardeen
	:inherited-members:
	:members:
	
Steady-State Models API
-----------------------

.. autoclass:: tripletpairs.kineticmodelling.steadystatemodels.MerrifieldExplicit1TT
	:inherited-members:
	:members:
	
.. autoclass:: tripletpairs.kineticmodelling.steadystatemodels.Merrifield
	:inherited-members:
	:members:
	
.. autoclass:: tripletpairs.kineticmodelling.steadystatemodels.Bardeen
	:inherited-members:
	:members: