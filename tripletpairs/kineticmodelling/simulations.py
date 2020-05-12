from tripletpairs.spin import SpinHamiltonian
from tripletpairs.toolkit import convolve_irf, integrate_between
import numpy as np

class KineticSimulation:
    """
    A wrapper class for performing general rate-model-based simulations.
    
    Averaging over certain spin Hamiltonian parameters is taken care of. This
    allows the user to simulate the effect of randomly oriented molecules
    (amorphous morphology) or randomly oriented crystallites
    (polycrystalline morphology). Refer to the examples for usage.
    
    Parameters
    ----------
    kinetic_model : model instance
        An instance of one of the kinetic models, either steady-state or time-resolved.
        
    Attributes
    ----------
    spin_hamiltonian : tripletpairs.spin.SpinHamiltonian
        An instance of the :class:`tripletpairs.spin.SpinHamiltonian` class.
    kinetic_model : model instance
        An instance of one of the kinetic models, either steady-state or time-resolved.
    state_populations : dict
        Created by :meth:`simulate_state_populations`. Keys are excited-state names, values are numpy arrays. Rows are times and columns are magnetic field strengths.
    eigenvalues : numpy.ndarray
        Eigenvalues of the triplet-pair states, optionally created by :meth:`simulate_state_populations`. Rows are the 9 states, columns are magnetic field strengths.
    
    """
    
    def __init__(self, kinetic_model):
        self.kinetic_model = kinetic_model
        self.spin_hamiltonian = SpinHamiltonian()
        
    def set_spin_hamiltonian_parameters(self, J, X, rAB, D, E, alpha, beta, gamma, B, theta, phi):
        r"""
        Input the parameters for the quantum-mechanical part of the simulation.
        
        Refer to :class:`tripletpairs.spin.SpinHamiltonian` for details.

        Parameters
        ----------
        J : float or numpy.ndarray
            Value of the intertriplet exchange energy in eV. If given as a 1D array, the simulation will computed for each value and the results averaged.
        X : float
            Value of the intertriplet dipole-dipole coupling strength in eV.
        rAB : 3-tuple of float
            Centre-of-mass vector (x, y, z) between the two molecules comprising the triplet-pair. Must be given in the molecular coordinate system of molecule A.
        D : float
            Value of the intratriplet zero-field splitting parameter D in eV.
        E : float
            Value of the intratriplet zero-field splitting parameter E in eV.
        alpha : float or numpy.ndarray
            Euler angle :math:`\alpha` that would rotate molecule A onto molecule B using the ZX'Z'' convention, in radians. Must be calculated in the molecular coordinate system of molecule A. If given as a 1D array, the simulation will computed for each value and the results averaged.
        beta : float or numpy.ndarray
            Euler angle :math:`\beta` that would rotate molecule A onto molecule B using the ZX'Z'' convention, in radians. Must be calculated in the molecular coordinate system of molecule A. If given as a 1D array, the simulation will computed for each value and the results averaged.
        gamma : float or numpy.ndarray
            Euler angle :math:`\gamma` that would rotate molecule A onto molecule B using the ZX'Z'' convention, in radians. Must be calculated in the molecular coordinate system of molecule A. If given as a 1D array, the simulation will computed for each value and the results averaged.
        B : float or numpy.ndarray
            Value of the external magnetic field strength in Tesla. If given as a 1D array, the simulation will be performed at each value in the array thereby allowing subsequent calculation of magnetic field effects.
        theta : float or numpy.ndarray
            Spherical polar angle :math:`\theta` defining the orientation of the magnetic field in the molecular coordinate system of molecule A. If given as a 1D array, the simulation will computed for each value and the results averaged.
        phi : float or numpy.ndarray
            Spherical polar angle :math:`\phi` defining the orientation of the magnetic field in the molecular coordinate system of molecule A. If given as a 1D array, the simulation will computed for each value and the results averaged.
            
        Returns
        -------
        None.

        """
        self._J_range = np.atleast_1d(J)
        self.spin_hamiltonian.X = X
        self.spin_hamiltonian.rAB = rAB
        self.spin_hamiltonian.D = D
        self.spin_hamiltonian.E = E
        self._alpha_range = np.atleast_1d(alpha)
        self._beta_range = np.atleast_1d(beta)
        self._gamma_range = np.atleast_1d(gamma)
        self._B_range = np.atleast_1d(B)
        self._theta_range = np.atleast_1d(theta)
        self._phi_range = np.atleast_1d(phi)
        return
    
    def convolve_populations_with_irf(self, fwhm, shift_to_zero=None):
        """
        Convolve simulation results with a gaussian IRF.

        Parameters
        ----------
        fwhm : float
            Full width half maximum of the IRF in same time units as t.
        shift_max_to_zero : str, optional
            If specified, the results will be shifted such that at t = 0, the specified state is maximal. The default is None.

        Returns
        -------
        None.

        """
        new_state_populations = {}
        t, y = convolve_irf(self.times, self.state_populations[list(self.state_populations.keys())[0]][:, 0], fwhm)
        for state, population in self.state_populations.items():
            new_state_populations[state] = np.zeros((len(t), len(self._B_range)))
            for i in range(len(self._B_range)):
                t, y = convolve_irf(self.times, population[:, i], fwhm)
                new_state_populations[state][:, i] = y
        if shift_to_zero is not None:
            idx = np.argmax(self.state_populations[shift_to_zero][:, 0])
            t -= t[idx]
        self.times = t
        self.state_populations = new_state_populations
        return
    
    def simulate_state_populations(self, states, return_eigenvalues=False):
        """
        Perform the simulation.
        
        The results are stored in :attr:`KineticSimulation.state_populations`.

        Parameters
        ----------
        states : list of str
            The excited-states to evaluate. These will be the keys of :attr:`KineticSimulation.state_populations`.
        return_eigenvalues : bool, optional
            If True, also calculate the eigenvalues of the triplet-pair states. The default is False.

        Returns
        -------
        None.

        """
        if return_eigenvalues:
            self.eigenvalues = np.zeros((9, len(self._B_range)))
        else:
            self.eigenvalues = None
        
        if self.kinetic_model._time_resolved:
            self.kinetic_model._calculate_time_axis()
            self.times = self.kinetic_model.t
            self.state_populations = dict(zip(states, [np.zeros((len(self.times), len(self._B_range))) for element in range(len(states))]))
        else:
            self.state_populations = dict(zip(states, [np.zeros((1, len(self._B_range))) for element in range(len(states))]))

        for i, B in enumerate(self._B_range):    
            state_populations_i, eigenvalues_i = self._simulate_average(B, states, self.kinetic_model._time_resolved, return_eigenvalues)
            
            for state in states:
                self.state_populations[state][:, i] = state_populations_i[state]
                
            if return_eigenvalues:
                self.eigenvalues[:, i] = eigenvalues_i
                
        return
    
    def calculate_mfe(self, state='S1', time_range=None):
        r"""
        Compute the magnetic field effect for a given excited-state.

        Parameters
        ----------
        state : str, optional
            The name of the excited-state to use. The default is 'S1'.
        time_range : 2-tuple of float, optional
            The times between which to evaluate the MFE, for time-resolved simulations. The default is None.

        Raises
        ------
        ValueError
            If a time_range is specified for a non-time-resolved simulation or vice versa.

        Returns
        -------
        mfe : numpy.ndarray
            The calculated magnetic field effect, evaluated as :math:`\Delta PL/PL`.

        """
        if self.state_populations[state].shape[0] == 1 and time_range is not None:
            raise ValueError('time_range can only be specified for time-resolved simulations')
        elif self.state_populations[state].shape[0] > 1 and time_range is None:
            raise ValueError('time_range must be specified for time-resolved simulations')
        
        state_population = self.state_populations[state]
        if time_range is not None:
            t1, t2 = time_range
            mask = ((self.times >= t1) & (self.times <= t2))
            state_population = state_population[mask, :]
            state_population = np.trapz(state_population, x=self.times[mask], axis=0)/(t2-t1)
        
        state_population = np.squeeze(state_population)
        
        mfe = (state_population-state_population[0])/state_population[0]
        
        return mfe 
            
    def _simulate_average(self, B, states, time_resolved, return_eigenvalues):
        
        if return_eigenvalues:
            eigenvalues = np.zeros(9)
        else:
            eigenvalues = None
            
        if time_resolved:
            state_populations = dict(zip(states, [np.zeros_like(self.kinetic_model.t) for element in range(len(states))]))
        else:
            state_populations = dict(zip(states, np.zeros(len(states))))
        
        self.spin_hamiltonian.calculate_exchange_hamiltonian()
        self.spin_hamiltonian.calculate_zerofield_hamiltonian_single_molecule()
        self.spin_hamiltonian.calculate_zerofield_hamiltonian_molecule_A()
        self.spin_hamiltonian.calculate_dipoledipole_hamiltonian()

        counter = 1
        for alpha in self._alpha_range:
            for beta in self._beta_range:
                for gamma in self._gamma_range:
                    
                    self.spin_hamiltonian.alpha, self.spin_hamiltonian.beta, self.spin_hamiltonian.gamma = alpha, beta, gamma
                    self.spin_hamiltonian.calculate_zerofield_hamiltonian_molecule_B()
                    
                    for theta in self._theta_range:
                        for phi in self._phi_range:
                            
                            self.spin_hamiltonian.theta, self.spin_hamiltonian.phi = theta, phi
                            
                            for J in self._J_range:
                                
                                self.spin_hamiltonian.J = J
                                self.spin_hamiltonian.B = B
                                self.spin_hamiltonian.calculate_zeeman_hamiltonian()
                                self.spin_hamiltonian.calculate_hamiltonian()
                                self.spin_hamiltonian.calculate_eigenstates()
                                self.spin_hamiltonian.calculate_cslsq()
                                self.kinetic_model.cslsq = self.spin_hamiltonian.cslsq
                                
                                self.kinetic_model.simulate()
                                   
                                if return_eigenvalues:
                                    eigenvalues = (eigenvalues*(counter-1)+self.spin_hamiltonian.eigenvalues)/counter
                                    
                                for state in states:
                                    state_populations[state] = (state_populations[state]*(counter-1)+self.kinetic_model.simulation_results[state])/counter
                                    
                                counter += 1
        return state_populations, eigenvalues
                