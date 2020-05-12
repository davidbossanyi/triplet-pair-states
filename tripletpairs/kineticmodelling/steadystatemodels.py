from ._base import SteadyStateModel
import numpy as np


class MerrifieldExplicit1TT(SteadyStateModel):
    r"""
    A class for steady-state simulations using a modified version of Merrifield's model.
    
    The model explicity includes the :math:`^1(TT)` state, and allows for decay
    of a single triplet in a :math:`(T..T)` state.
    
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
    kSF : float
        Rate constant for :math:`S_1\rightarrow ^1(TT)`. Units of per time.
    k_SF : float
        Rate constant for :math:`^1(TT)\rightarrow S_1`. Units of per time.
    kHOP : float
        Rate constant for :math:`^1(TT)\rightarrow (T..T)`. Units of per time.
    k_HOP : float
        Rate constant for :math:`(T..T)\rightarrow ^1(TT)`. Units of per time.
    kHOP2 : float
        Rate constant for :math:`(T..T)\rightarrow2\times T_1`. Units of per time.
    kTTA : float
        Rate constant for :math:`2\times T_1\rightarrow (T..T)`. Units of volume per time.
    kSNR : float
        Rate constant for the decay of :math:`S_1`. Units of per time.
    kTTNR : float
        Rate constant for the decay of :math:`^1(TT)`. Units of per time.
    kTNR : float
        Rate constant for the decay of :math:`T_1`, or one of the triplets in :math:`(T..T)`. Units of per time.
    cslsq : numpy.ndarray
        1D array containing the overlap factors between the 9 :math:`(T..T)` states and the singlet.
    simulation_results : dict
        Produced by :meth:`simulate`. Keys are the excited-state names (str), values the simulated populations (float).
        
    """
    
    def __init__(self):
        super().__init__()
        # metadata
        self.model_name = 'MerrifieldExplicit1TT'
        self._number_of_states = 12
        self.states = ['S1', 'TT', 'T_T_total', 'T1']
        self.rates = ['kSF', 'k_SF', 'kHOP', 'k_HOP', 'kHOP2', 'kTTA', 'kSNR', 'kTTNR', 'kTNR']
        self._allowed_initial_states = {'S1', 'TT', 'T1'}
        self._initial_state_mapping = {'S1': 0, 'TT': 1, 'T1': -1}
        # rates between excited states
        self.kSF = 20.0
        self.k_SF = 0.03
        self.kHOP = 0.067
        self.k_HOP = 2.5e-4
        self.kHOP2 = 1e-5
        self.kTTA = 1e-18
        # rates of decay
        self.kSNR = 0.1
        self.kTTNR = 0.067
        self.kTNR = 1e-5
        # cslsq values
        self.cslsq = (1/9)*np.ones(9)
        
    def simulate(self):
        """
        Perform the simulation.

        Returns
        -------
        None.

        """
        self._set_generation_rates()
        self._calculate_intermediate_parameters()
        self._calculate_T1()
        self._calculate_other_states()
        self._wrap_simulation_results()
        return
    
    def _set_generation_rates(self):
        self._check_initial_weighting()
        self._set_initial_condition()
        self._GS = self._y0[self._initial_state_mapping['S1']]
        self._GTT = self._y0[self._initial_state_mapping['TT']]
        self._GT = self._y0[self._initial_state_mapping['T1']]
        return
        
    def _calculate_intermediate_parameters(self):
        self._bT = self.kTNR
        self._bST = self.kHOP*np.sum(((self.kTNR+2*self.kHOP2)*self.cslsq)/(self.k_HOP*self.cslsq+self.kHOP2+self.kTNR))
        self._aT = (self.kTTA/9)*np.sum((self.kTNR+2*self.k_HOP*self.cslsq)/(self.k_HOP*self.cslsq+self.kHOP2+self.kTNR))
        self._GTp = self._GT
        self._bS = ((self.k_SF*self.kSNR)/(self.kSF+self.kSNR))+self.kTTNR+self.kHOP*np.sum(((self.kTNR+self.kHOP2)*self.cslsq)/(self.k_HOP*self.cslsq+self.kHOP2+self.kTNR))
        self._GSp = ((self.kSF*self._GS)/(self.kSF+self.kSNR)) + self._GTT
        self._aS = (self.kTTA/9)*np.sum((self.k_HOP*self.cslsq)/(self.k_HOP*self.cslsq+self.kHOP2+self.kTNR))
        return
    
    def _calculate_T1(self):
        a = self._bS*self._aT-self._bST*self._aS
        b = self._bS*self._bT
        c = -1*(self._bST*self._GSp+self._bS*self._GTp)
        T1 = self._quadratic_formula(a, b, c)[1]
        self._check_root(T1)
        self.T1 = T1
        return
    
    def _calculate_other_states(self):
        self.TT = (self._GSp/self._bS)+(self._aS/self._bS)*self.T1*self.T1
        self.S1 = (self.k_SF/(self.kSF+self.kSNR))*self.TT+(self._GS/(self.kSF+self.kSNR))
        self.T_T_total = 0
        for i in range(9):
            T_T_i = (self.kHOP*self.cslsq[i]*self.TT+(self.kTTA/9)*self.T1*self.T1)/(self.k_HOP*self.cslsq[i]+self.kHOP2+self.kTNR)
            self.T_T_total += T_T_i
        return
    
    def _wrap_simulation_results(self):
        self.simulation_results = dict(zip(self.states, [self.S1, self.TT, self.T_T_total, self.T1]))
        return
    
    
class Merrifield(SteadyStateModel):
    r"""
    A class for steady-state simulations using Merrifield's model.
    
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
    kSF : float
        Rate constant for :math:`S_1\rightarrow (TT)`. Units of per time.
    k_SF : float
        Rate constant for :math:`(TT)\rightarrow S_1`. Units of per time.
    kDISS : float
        Rate constant for :math:`(TT)\rightarrow2\times T_1`. Units of per time.
    kTTA : float
        Rate constant for :math:`2\times T_1\rightarrow (TT)`. Units of volume per time.
    kSNR : float
        Rate constant for the decay of :math:`S_1`. Units of per time.
    kTTNR : float
        Rate constant for the decay of :math:`(TT)`. Units of per time.
    kTNR : float
        Rate constant for the decay of :math:`T_1`. Units of per time.
    cslsq : numpy.ndarray
        1D array containing the overlap factors between the 9 :math:`(T..T)` states and the singlet.
    simulation_results : dict
        Produced by :meth:`simulate`. Keys are the excited-state names (str), values the simulated populations (float).
        
    """
    
    def __init__(self):
        super().__init__()
        # metadata
        self.model_name = 'Merrifield'
        self._number_of_states = 11
        self.states = ['S1', 'TT_bright', 'TT_total', 'T1']
        self.rates = ['kSF', 'k_SF', 'kDISS', 'kTTA', 'kSNR', 'kTTNR', 'kTNR']
        # rates between excited states
        self.kSF = 20.0
        self.k_SF = 0.03
        self.kDISS = 0.067
        self.kTTA = 1e-18
        # rates of decay
        self.kSNR = 0.1
        self.kTTNR = 0.067
        self.kTNR = 1e-5
        # cslsq values
        self.cslsq = (1/9)*np.ones(9)
        
    def simulate(self):
        """
        Perform the simulation.

        Returns
        -------
        None.

        """
        self._set_generation_rates()
        self._calculate_intermediate_parameters()
        self._calculate_T1()
        self._calculate_other_states()
        self._wrap_simulation_results()
        return
    
    def _set_generation_rates(self):
        self._check_initial_weighting()
        self._set_initial_condition()
        self._GS = self._y0[self._initial_state_mapping['S1']]
        self._GT = self._y0[self._initial_state_mapping['T1']]
        return
        
    def _calculate_intermediate_parameters(self):
        self._bT = self.kTNR
        self._bST = 2*self.kSF*np.sum((self.kDISS*self.cslsq)/(self.k_SF*self.cslsq+self.kDISS+self.kTTNR))
        self._aT = (2*self.kTTA/9)*np.sum(((self.k_SF*self.cslsq)+self.kTTNR)/(self.k_SF*self.cslsq+self.kDISS+self.kTTNR))
        self._GTp = self._GT
        self._bS = self.kSNR+self.kSF*np.sum(((self.kDISS+self.kTTNR)*self.cslsq)/(self.k_SF*self.cslsq+self.kDISS+self.kTTNR))
        self._GSp = self._GS
        self._aS = (self.kTTA/9)*np.sum((self.k_SF*self.cslsq)/(self.k_SF*self.cslsq+self.kDISS+self.kTTNR))
        return
    
    def _calculate_T1(self):
        a = self._bS*self._aT-self._bST*self._aS
        b = self._bS*self._bT
        c = -1*(self._bST*self._GSp+self._bS*self._GTp)
        T1 = self._quadratic_formula(a, b, c)[1]
        self._check_root(T1)
        self.T1 = T1
        return
    
    def _calculate_other_states(self):
        self.S1 = (self._GSp/self._bS)+(self._aS/self._bS)*self.T1*self.T1
        self.TT_total, self.TT_bright = 0, 0
        for i in range(9):
            TT_i = (self.kSF*self.cslsq[i]*self.S1+(self.kTTA/9)*self.T1*self.T1)/(self.k_SF*self.cslsq[i]+self.kDISS+self.kTTNR)
            self.TT_total += TT_i
            self.TT_bright += self.cslsq[i]*TT_i
        return

    def _wrap_simulation_results(self):
        self.simulation_results = dict(zip(self.states, [self.S1, self.TT_bright, self.TT_total, self.T1]))
        return


class Bardeen(SteadyStateModel):
    r"""
    A class for steady-state simulations using a modified version of Merrifield's model.
    
    The model does not include free triplets. Instead Merrifield's :math:`(TT)`
    states can separate to form 9 :math:`(T..T)` states which can undergo spin
    relaxation. This is an approximation to triplet-diffusion in the limit of
    low excitation density.
    
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
    kSF : float
        Rate constant for :math:`S_1\rightarrow (TT)`. Units of per time.
    k_SF : float
        Rate constant for :math:`(TT)\rightarrow S_1`. Units of per time.
    kHOP : float
        Rate constant for :math:`(TT)\rightarrow (T..T)`. Units of per time.
    k_HOP : float
        Rate constant for :math:`(T..T)\rightarrow (TT)`. Units of per time.
    kRELAX : float
        Rate constant for mixing between the :math:`(T..T)` states. Units of per time.
    kSNR : float
        Rate constant for the decay of :math:`S_1`. Units of per time.
    kTTNR : float
        Rate constant for the decay of :math:`(TT)`. Units of per time.
    kSPIN : float
        Rate constant for the decay of :math:`(T..T)`. Units of per time.
    cslsq : numpy.ndarray
        1D array containing the overlap factors between the 9 :math:`(T..T)` states and the singlet.
    simulation_results : dict
        Produced by :meth:`simulate`. Keys are the excited-state names (str), values the simulated populations (float).
        
    """
    
    def __init__(self):
        super().__init__()
        # metadata
        self.model_name = 'Bardeen'
        self._number_of_states = 19
        self.states = ['S1', 'TT_bright', 'TT_total', 'T_T_total']
        self.rates = ['kSF', 'k_SF', 'kHOP', 'k_HOP', 'kRELAX', 'kSNR', 'kTTNR', 'kSPIN']
        self._allowed_initial_states = {'S1'}
        self._initial_state_mapping = {'S1': 0}
        # rates between excited states
        self.kSF = 20.0
        self.k_SF = 0.03
        self.kHOP = 0.067
        self.k_HOP = 2.5e-4
        # spin relaxation
        self.kRELAX = 0.033
        # rates of decay
        self.kSNR = 0.1
        self.kTTNR = 0.067
        self.kSPIN = 2.5e-4
        # cslsq values
        self.cslsq = (1/9)*np.ones(9)
        
    def simulate(self):
        """
        Perform the simulation.

        Returns
        -------
        None.

        """
        self._set_generation_rates()
        self._generate_generation_rate_matrix()
        self._generate_rate_equation_matrix()
        y = np.linalg.solve(self._rem, self._grm)
        self._unpack_simulation(y)
        self._wrap_simulation_results()
        return
    
    def _set_generation_rates(self):
        self._check_initial_weighting()
        self._set_initial_condition()
        self._GS = self._y0[self._initial_state_mapping['S1']]
        return
    
    def _generate_rate_equation_matrix(self):
        self._rem = np.zeros((self._number_of_states, self._number_of_states))
        self._rem[0, 0] = -1*(self.kSNR+self.kSF*np.sum(self.cslsq))
        self._rem[10:, 10:] = (1/8)*self.kRELAX
        for i in range(9):
            self._rem[0, i+1] = self.k_SF*self.cslsq[i]
            self._rem[i+1, 0] = self.kSF*self.cslsq[i]
            self._rem[i+1, i+1] = -1*((self.k_SF+self.kTTNR)*self.cslsq[i]+self.kHOP)
            self._rem[i+1, i+10] = self.k_HOP
            self._rem[i+10, i+1] = self.kHOP
            self._rem[i+10, i+10] = -1*(self.kHOP+self.kSPIN+self.kRELAX)
        return
    
    def _generate_generation_rate_matrix(self):
        self._grm = np.zeros((self._number_of_states, 1))
        self._grm[0, 0] = self._GS
        return
    
    def _unpack_simulation(self, y):
        self.S1 = y[0, 0]
        self.TT_bright, self.TT_total, self.T_T_total = 0, 0, 0
        for i in range(9):
            self.TT_bright += self.cslsq[i]*y[i+1, 0]
            self.TT_total += y[i+1, 0]
            self.T_T_total += y[i+10, 0]
        return
    
    def _wrap_simulation_results(self):
        self.simulation_results = dict(zip(self.states, [self.S1, self.TT_bright, self.TT_total, self.T_T_total]))
        return 
       