from tripletpairs.spin import SpinHamiltonian
import numpy as np

class KineticSimulation:
    
    def __init__(self, kinetic_model):
        self.kinetic_model = kinetic_model
        self.spin_hamiltonian = SpinHamiltonian()
        
    def set_spin_hamiltonian_parameters(self, J, X, rAB, D, E, alpha, beta, gamma, B, theta, phi):
        self.J_range = np.atleast_1d(J)
        self.spin_hamiltonian.X = X
        self.spin_hamiltonian.rAB = rAB
        self.spin_hamiltonian.D = D
        self.spin_hamiltonian.E = E
        self.alpha_range = np.atleast_1d(alpha)
        self.beta_range = np.atleast_1d(beta)
        self.gamma_range = np.atleast_1d(gamma)
        self.B_range = np.atleast_1d(B)
        self.theta_range = np.atleast_1d(theta)
        self.phi_range = np.atleast_1d(phi)
        return
    
    def simulate_state_populations(self, states, time_resolved, return_eigenvalues=False):
        
        if return_eigenvalues:
            self.eigenvalues = np.zeros(9, len(self.B_range))
        else:
            self.eigenvalues = None
        
        if time_resolved:
            self.times = self.kinetic_model.t
            self.state_populations = dict(zip(states, [np.zeros((len(self.kinetic_model.t), len(self.B_range))) for element in range(len(states))]))
        else:
            self.state_populations = dict(zip(states, [np.zeros((1, len(self.B_range))) for element in range(len(states))]))

        for i, B in enumerate(self.B_range):    
            state_populations_i, eigenvalues_i = self._simulate_average(B, states, time_resolved, return_eigenvalues)
            
            for state in states:
                self.state_populations[state][:, i] = state_populations_i[state]
                
            if return_eigenvalues:
                self.eigenvalues[:, i] = eigenvalues_i
                
        return
    
    def calculate_mfe(self, state='S1', time_range=None):
        
        if self.state_populations[state].shape[0] == 1 and time_range is not None:
            raise ValueError('time_range can only be specified for time-resolved simulations')
        elif self.state_populations[state].shape[0] > 1 and time_range is None:
            raise ValueError('time_range must be specified for time-resolved simulations')
        
        if time_range is None:
            state_population = self.state_populations[state]
        else:
            t1, t2 = time_range
            mask = ((self.kinetic_model.t >= t1) & (self.kinetic_model.t <= t2))
            state_population = self.state_populations[state][mask, :]
            state_population = np.trapz(state_population, x=self.kinetic_model.t[mask], axis=0)/(t2-t1)
        
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
        for alpha in self.alpha_range:
            for beta in self.beta_range:
                for gamma in self.gamma_range:
                    
                    self.spin_hamiltonian.alpha, self.spin_hamiltonian.beta, self.spin_hamiltonian.gamma = alpha, beta, gamma
                    self.spin_hamiltonian.calculate_zerofield_hamiltonian_molecule_B()
                    
                    for theta in self.theta_range:
                        for phi in self.phi_range:
                            
                            self.spin_hamiltonian.theta, self.spin_hamiltonian.phi = theta, phi
                            
                            for J in self.J_range:
                                    
                                self.spin_hamiltonian.B = B
                                self.spin_hamiltonian.calculate_zeeman_hamiltonian()
                                self.spin_hamiltonian.calculate_hamiltonian()
                                self.spin_hamiltonian.calculate_eigenstates()
                                self.spin_hamiltonian.calculate_cslsq()
                                self.kinetic_model.cslsq = self.spin_hamiltonian.cslsq
                                
                                if time_resolved:
                                    self.kinetic_model.simulate_time_resolved()
                                else:
                                    self.kinetic_model.simulate_steady_state()
                                   
                                if return_eigenvalues:
                                    eigenvalues = (eigenvalues*(counter-1)+self.spin_hamiltonian.eigenvalues)/counter
                                    
                                for state in states:
                                    state_populations[state] = (state_populations[state]*(counter-1)+self.kinetic_model.simulation_results[state])/counter
                                    
                                counter += 1
        return state_populations, eigenvalues
                