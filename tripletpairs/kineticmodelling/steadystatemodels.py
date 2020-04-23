from ._base import SteadyStateModel
import numpy as np


class MerrifieldExplicit1TT(SteadyStateModel):
    """
    This is basically Merrifields model, but explicitly separating 
    the 1(TT) from S1 and (T..T).
    """
    def __init__(self):
        super().__init__()
        # metadata
        self.model_name = 'MerrifieldExplicit1TT'
        self.number_of_states = 12
        self.states = ['S1', 'TT', 'T_T_total', 'T1']
        self.rates = ['kSF', 'k_SF', 'kHOP', 'k_HOP', 'kHOP2', 'KTTA', 'kSNR', 'kTTNR', 'kTNR']
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
        self._set_generation_rates()
        self._calculate_intermediate_parameters()
        self._calculate_T1()
        self._calculate_other_states()
        self._wrap_simulation_results()
        return
        
    def _calculate_intermediate_parameters(self):
        self._bT = self.kTNR
        self._bST = self.kHOP*np.sum(((self.kTNR+2*self.kHOP2)*self.cslsq)/(self.k_HOP*self.cslsq+self.kHOP2+self.kTNR))
        self._aT = (self.kTTA/9)*np.sum((self.kTNR+2*self.k_HOP*self.cslsq)/(self.k_HOP*self.cslsq+self.kHOP2+self.kTNR))
        self._GTp = self._GT
        self._bS = ((self.k_SF*self.kSNR)/(self.kSF+self.kSNR))+self.kTTNR+self.kHOP*np.sum(((self.kTNR+self.kHOP2)*self.cslsq)/(self.k_HOP*self.cslsq+self.kHOP2+self.kTNR))
        self._GSp = (self.kSF*self._GS)/(self.kSF+self.kSNR)
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
    """
    The standard Merrifield model.
    """
    def __init__(self):
        super().__init__()
        # metadata
        self.model_name = 'Merrifield'
        self.number_of_states = 11
        self.states = ['S1', 'TT_bright', 'TT_total', 'T1']
        self.rates = ['kSF', 'k_SF', 'kDISS', 'KTTA', 'kSNR', 'kTTNR', 'kTNR']
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
        self._set_generation_rates()
        self._calculate_intermediate_parameters()
        self._calculate_T1()
        self._calculate_other_states()
        self._wrap_simulation_results()
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
    """
    Variation of Merrifield model presented by Bardeen and co-workers. It is
    valid at low fluence (no TTA) and adds triplet diffusion in a crude
    fashion.
    """
    def __init__(self):
        super().__init__()
        # metadata
        self.model_name = 'Bardeen'
        self.number_of_states = 19
        self.states = ['S1', 'TT_bright', 'TT_total', 'T_T_total']
        self.rates = ['kSF', 'k_SF', 'kHOP', 'k_HOP', 'kRELAX', 'kSNR', 'kTTNR', 'kSPIN']
        self._allowed_initial_states = {'S1'}
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
        self._set_generation_rates()
        self._generate_generation_rate_matrix()
        self._generate_rate_equation_matrix()
        y = np.linalg.solve(self._rem, self._grm)
        self._unpack_simulation(y)
        self._wrap_simulation_results()
        return
    
    def _generate_rate_equation_matrix(self):
        self._rem = np.zeros((self.number_of_states, self.number_of_states))
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
        self._grm = np.zeros((self.number_of_states, 1))
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
       