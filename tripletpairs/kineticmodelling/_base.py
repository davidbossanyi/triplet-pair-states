import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class RateModel:
    """
    Base class for all rate models.
    
    Attributes
    ----------
    states : list of str
        The names of the excited state species.
    rates : list of str
        The names of the different rate constants in the model.
    model_name : str
        The name of the model.
    G : float
        The initial exciton density. Units of per volume.
    initial_weighting : dict
        Dictionary of (str, float) pairs. Key is the state name (str) and value is its initial weight (float).
    """
    
    def __init__(self):
        self._number_of_states = 2
        self.states = ['S1', 'T1']
        self.rates = []
        self.model_name = 'base'
        self._time_resolved = True
        self.G = 1e17
        self._allowed_initial_states = {'S1', 'T1'}
        self._initial_state_mapping = {'S1': 0, 'T1': -1}
        self.initial_weighting = {'S1': 1}
        
    def _check_initial_weighting(self):
        for starting_state in self.initial_weighting.keys():
            if starting_state not in self._allowed_initial_states:
                raise ValueError('invalid state {0} in initial_weighting'.format(starting_state))
            if self.initial_weighting[starting_state] < 0:
                raise ValueError('weightings must be positive')
        return
            
    def _set_initial_condition(self):
        self._y0 = np.zeros(self._number_of_states)
        total_weights = np.sum(np.array(list(self.initial_weighting.values())))
        for key in self.initial_weighting.keys():
            idx = self._initial_state_mapping[key]
            weight = self.initial_weighting[key]/total_weights
            self._y0[idx] = weight*self.G
        return


class TimeResolvedModel(RateModel):
    """
    Base class for all time-resolved models.
    
    Attributes
    ----------
    states : list of str
        The names of the excited state species.
    rates : list of str
        The names of the different rate constants in the model.
    model_name : str
        The name of the model.
    G : float
        The initial exciton density. Units of per volume.
    initial_weighting : dict
        Dictionary of (str, float) pairs. Key is the state name (str) and value is its initial weight (float).
    t_step : float
        The first time step taken by the simulation, thereafter the step will increase geometrically.
    t_end : float
        The last time point in the simulation.
    num_points : int
        The number of time points to compute the simulation at.
    
    """
    
    def __init__(self):
        super().__init__()
        self.t_step = 0.0052391092278624
        self.t_end = 1e6
        self.num_points = 10000
        return
    
    def _calculate_time_axis(self):
        self.t = np.geomspace(self.t_step, self.t_end+self.t_step, self.num_points)-self.t_step
        self.t[0] = 0
        return
    
    def view_timepoints(self):
        """
        Produce a plot showing the distribution of times, print the first and last 5.

        Returns
        -------
        None.

        """
        self._calculate_time_axis()
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.semilogx(self.t, np.ones_like(self.t), 'bx')
        plt.show()
        print('\n')
        for t in self.t[0:5]:
            print(t)
        print('\n')
        for t in self.t[-5:]:
            print(t)
        return
    
    def _rate_equations(self, y, t):
        """Will be modified for each rate model."""
        return np.ones(self._number_of_states+1)
        
    def _initialise_simulation(self):
        self._calculate_time_axis()
        self._check_initial_weighting()
        self._set_initial_condition()
        return
    
    def simulate(self):
        """
        Perform the simulation.

        Returns
        -------
        None.

        """
        self._initialise_simulation()
        y = odeint(lambda y, t: self._rate_equations(y, t), self._y0, self.t)
        self._unpack_simulation(y)
        return
        
    def _unpack_simulation(self, y):
        """Will be modified for each rate model."""
        pass

    
class SteadyStateModel(RateModel):
    """
    Base class for all steady-state models.
    
    Attributes
    ----------
    states : list of str
        The names of the excited state species.
    rates : list of str
        The names of the different rate constants in the model.
    model_name : str
        The name of the model.
    initial_weighting : dict
        Dictionary of (str, float) pairs. Key is the state name (str) and value is its initial weight (float).
    G : float
        The exciton generation rate for :attr:`initial_state`. Units of per volume per time.
    
    """
    
    def __init__(self):
        super().__init__()
        self._time_resolved = False
        self.G = 2.7e13
        return
    
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
            