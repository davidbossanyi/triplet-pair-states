import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve


class KineticModelBase:
    """
    Base class for all kinetic models.
    """
    def __init__(self):
        self.number_of_states = 2
        self.states = ['S1', 'T1']
        self.rates = ['kGEN']
        self.model_name = 'base'
        self._allowed_initial_states = {'S1, T1'}
        self.t = np.logspace(-5, 5, 10000)
        self.initial_state = 'S1'
        self.G = 1e17
        self.kGEN = 1.8
        return
    
    def _rate_equations(self, y, t):
        return np.ones(self.number_of_states+1)
    
    def _set_generation_rates(self, time_resolved=True):
        if time_resolved:
            self._GS = 0
            self._GT = 0
            if self.initial_state == 'S1':
                self._kGENS = self.kGEN
                self._kGENT = 0
            elif self.initial_state == 'T1':
                self._kGENS = 0
                self._kGENT = self.kGEN
            else:
                raise ValueError('initial_state attribute must be one of {0}'.format(self._allowed_initial_states))
        else:
            self._kGENS = 0
            self._kGENT = 0
            if self.initial_state == 'S1':
                self._GS = self.G
                self._GT = 0
            elif self.initial_state == 'T1':
                self._GS = 0
                self._GT = self.G
            else:
                raise ValueError('initial_state attribute must be one of {0}'.format(self._allowed_initial_states))
        return
            
    def _set_initial_condition(self, time_resolved=True):
        if time_resolved:
            y0 = np.zeros(self.number_of_states+1)
            y0[0] = self.G
        else:
            y0 = self.G*np.ones(self.number_of_states+1)
            y0[0] = 0
        self.y0 = y0
        return
        
    def _initialise_simulation_tr(self):
        self._set_initial_condition(time_resolved=True)
        self._set_generation_rates(time_resolved=True)
        return
    
    def _initialise_simulation_ss(self):
        self._set_initial_condition(time_resolved=False)
        self._set_generation_rates(time_resolved=False)
        return
    
    def simulate_time_resolved(self):
        self._initialise_simulation_tr()
        y = odeint(lambda y, t: self._rate_equations(y, t), self.y0, self.t)
        self._unpack_simulation_tr(y)
        return
        
    def _unpack_simulation_tr(self, y):
        pass
    
    def simulate_steady_state(self):
        self._initialise_simulation_ss()
        y = fsolve(lambda y: self._rate_equations(y, 0), self.y0)
        self._unpack_simulation_ss(y)
        return
        
    def _unpack_simulation_ss(self, y):
        pass
    
    def get_population_between(self, array, t1, t2):
        """
        integrates the dynamics of array (e.g. model.S1) 
        between times t1 and t2
        """
        mask = ((self.t >= t1) & (self.t <= t2))
        t = self.t[mask]
        population = np.trapz(array[mask], x=t)/(t2-t1)
        return population
    
    def normalise_population_at(self, array, t):
        """
        normalises the dynamics of array (e.g. model.S1) at time t
        """
        idx = np.where((self.t-t)**2 == min((self.t-t)**2))[0][0]
        factor = array[idx]
        array = array/factor
        return array, factor
