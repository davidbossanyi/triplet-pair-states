import numpy as np
from scipy.integrate import odeint


class TimeResolvedModel:
    """
    Base class for all time-resolved models.
    """
    def __init__(self):
        self.number_of_states = 2
        self.states = ['S1', 'T1']
        self.rates = ['kGEN']
        self.model_name = 'base'
        self.time_resolved = True
        self._allowed_initial_states = {'S1', 'T1'}
        self.t = np.logspace(-5, 5, 10000)
        self.initial_state = 'S1'
        self.G = 1e17
        self.kGEN = 1.8
        return
    
    def _rate_equations(self, y, t):
        return np.ones(self.number_of_states+1)
    
    def _set_generation_rates(self):
        if self.initial_state not in self._allowed_initial_states:
            raise ValueError('initial_state attribute must be one of {0}'.format(self._allowed_initial_states))
        if self.initial_state == 'S1':
            self._kGENS = self.kGEN
            self._kGENT = 0
        elif self.initial_state == 'T1':
            self._kGENS = 0
            self._kGENT = self.kGEN
        else:
            raise ValueError('initial_state attribute must be one of {0}'.format(self._allowed_initial_states))
        return
            
    def _set_initial_condition(self):
        self.y0 = np.zeros(self.number_of_states+1)
        self.y0[0] = self.G
        return
        
    def _initialise_simulation(self):
        self._set_initial_condition()
        self._set_generation_rates()
        return
    
    def simulate(self):
        self._initialise_simulation()
        y = odeint(lambda y, t: self._rate_equations(y, t), self.y0, self.t)
        self._unpack_simulation(y)
        return
        
    def _unpack_simulation(self, y):
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

    
class SteadyStateModel:
    """
    Base class for all steady-state models.
    """
    def __init__(self):
        self.number_of_states = 2
        self.states = ['S1', 'T1']
        self.rates = []
        self.model_name = 'base'
        self.time_resolved = False
        self._allowed_initial_states = {'S1', 'T1', 'injection'}
        self.initial_state = 'S1'
        self.G = 2.7e13
        return
    
    def _set_generation_rates(self):
        if self.initial_state not in self._allowed_initial_states:
            raise ValueError('initial_state attribute must be one of {0}'.format(self._allowed_initial_states))
        if self.initial_state == 'S1':
            self._GS = self.G
            self._GT = 0
        elif self.initial_state == 'T1':
            self._GS = 0
            self._GT = self.G
        elif self.initial_state == 'injection':
            self._GS = self.G/4
            self._GT = 3*self.G/4
        else:
            raise ValueError('initial_state attribute must be one of {0}'.format(self._allowed_initial_states))
        return
    
    def simulate(self):
        pass
    
    @staticmethod
    def _quadratic_formula(a, b, c):
        det = np.sqrt(b*b - 4*a*c)
        return (-1*b-det)/(2*a), (-1*b+det)/(2*a)
    
    @staticmethod
    def _check_root(root):
        if not np.isreal(root):
            raise RuntimeError('no valid solution found')
        else:
            if root < 0:
                raise RuntimeError('no valid solution found')
            