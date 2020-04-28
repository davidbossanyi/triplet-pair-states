Spin Hamiltonian
================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
Introduction
------------

The Hamiltonian is the sum of the Zeeman, zero-field, intertriplet dipole-dipole and intertriplet exchange terms.
	
For details, refer to `this paper <https://doi.org/10.1021/acs.jpcc.6b04934>`_.

The class uses:

- :math:`\hbar=1`
- Tesla for magnetic field strength
- electronvolts for energies
- radians for angles
- ZX'Z'' convention for Euler angles
- the molecular system of molecule A throughout, which takes the x-axis parallel to the long molecular axis, the y-axis parallel to the short molecular axis and the z-axis perpendicular to the plane of the molecular backbone

The zero-field basis is used for the construction of the Hamiltonian:

.. math::

   \{|\phi_i\rangle\}=\{|xx\rangle, |xy\rangle, |xz\rangle, |yx\rangle, |yy\rangle, |yz\rangle, |zx\rangle, |zy\rangle, |zz\rangle\}

The :class:`tripletpairs.spin.SpinHamiltonian` can be used to compute the total spin Hamiltonian, its eigenvectors and eigenvalues and the overlap factors :math:`|C_S^l|^2` that are required for further simulations.

Example
-------

.. code-block:: python

   from tripletpairs.spin import SpinHamiltonian
   # set the parameters
   sh = SpinHamiltonian()
   sh.D = 5.1e-6
   sh.E = 1.6e-7
   sh.X = 1.1e-9
   sh.J = 0
   sh.rAB = (0.7636, -0.4460, 0.4669)
   sh.alpha, sh.beta, sh.gamma = 0, 0, 0
   sh.theta, sh.phi = 0, 0
   sh.B = 0
   # the individual parts of the hamiltonian must be calculated separately
   sh.calculate_exchange_hamiltonian()
   sh.calculate_zerofield_hamiltonian_single_molecule()
   sh.calculate_zerofield_hamiltonian_molecule_A()
   sh.calculate_dipoledipole_hamiltonian()
   sh.calculate_zerofield_hamiltonian_molecule_B()
   sh.calculate_zeeman_hamiltonian()
   sh.calculate_hamiltonian()
   # then the quantities of interest can be calculated
   sh.calculate_eigenstates()
   sh.calculate_cslsq()
   # and accessed as attributes: 
   cslsq = sh.cslsq
   eigenvalues = sh.eigenvalues
   eigenstates = sh.eigenstates
   # alternatively, calculate everything all at once
   sh.calculate_everything()

API
---

.. autoclass:: tripletpairs.spin.SpinHamiltonian
	:members: