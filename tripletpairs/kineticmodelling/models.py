from ._base import KineticModelBase
import numpy as np


class MerrifieldExplicit1TT(KineticModelBase):
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
        self.rates = ['kGEN', 'kSF', 'k_SF', 'kHOP', 'k_HOP', 'kHOP2', 'KTTA', 'kRELAX', 'kSNR', 'kSSA', 'kTTNR', 'kTNR']
        # rates between excited states
        self.kSF = 20.0
        self.k_SF = 0.03
        self.kHOP = 0.067
        self.k_HOP = 2.5e-4
        self.kHOP2 = 1e-5
        self.kTTA = 1e-18
        # spin relaxation
        self.kRELAX = 0
        # rates of decay
        self.kSNR = 0.1
        self.kSSA = 0
        self.kTTNR = 0.067
        self.kTNR = 1e-5
        # TTA channel
        self.TTA_channel = 1
        # cslsq values
        self.cslsq = (1/9)*np.ones(9)

    def _rate_equations(self, y, t):
        GS, S1, TT, T_T_1, T_T_2, T_T_3, T_T_4, T_T_5, T_T_6, T_T_7, T_T_8, T_T_9, T1 = y
        dydt = np.zeros(self.number_of_states+1)
        # GS
        dydt[0] = -(self._kGENS+self._kGENT)*GS
        # S1
        dydt[1] = self._GS + self._kGENS*GS - (self.kSNR+self.kSF)*S1 - self.kSSA*S1*S1 + self.k_SF*TT + self._kTTA_3*T1**2
        # TT
        dydt[2] = self.kSF*S1 - (self.k_SF+self.kTTNR+self.kHOP*np.sum(self.cslsq))*TT + self.k_HOP*(self.cslsq[0]*T_T_1+self.cslsq[1]*T_T_2+self.cslsq[2]*T_T_3+self.cslsq[3]*T_T_4+self.cslsq[4]*T_T_5+self.cslsq[5]*T_T_6+self.cslsq[6]*T_T_7+self.cslsq[7]*T_T_8+self.cslsq[8]*T_T_9) + self._kTTA_2*T1**2
        # T_T_1
        dydt[3] = self.kHOP*self.cslsq[0]*TT - (self.k_HOP*self.cslsq[0]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_1 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_2
        dydt[4] = self.kHOP*self.cslsq[1]*TT - (self.k_HOP*self.cslsq[1]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_2 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_3
        dydt[5] = self.kHOP*self.cslsq[2]*TT - (self.k_HOP*self.cslsq[2]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_3 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_4
        dydt[6] = self.kHOP*self.cslsq[3]*TT - (self.k_HOP*self.cslsq[3]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_4 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_5
        dydt[7] = self.kHOP*self.cslsq[4]*TT - (self.k_HOP*self.cslsq[4]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_5 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_6
        dydt[8] = self.kHOP*self.cslsq[5]*TT - (self.k_HOP*self.cslsq[5]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_6 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_7+T_T_8+T_T_9)
        # T_T_7
        dydt[9] = self.kHOP*self.cslsq[6]*TT - (self.k_HOP*self.cslsq[6]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_7 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_8+T_T_9)
        # T_T_8
        dydt[10] = self.kHOP*self.cslsq[7]*TT - (self.k_HOP*self.cslsq[7]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_8 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_9)
        # T_T_9
        dydt[11] = self.kHOP*self.cslsq[8]*TT - (self.k_HOP*self.cslsq[8]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_9 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8)
        # T1
        dydt[12] = self._GT + self._kGENT*GS + (self.kTNR+(2.0*self.kHOP2))*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9) - 2*self._kTTA_1*T1**2 - 2*self._kTTA_2*T1**2 - 2*self._kTTA_3*T1**2 - self.kTNR*T1
        #
        return dydt
    
    def _set_tta_rates(self):
        if self.TTA_channel == 1:  # this is T1 + T1 -> (T..T)
            self._kTTA_1 = self.kTTA
            self._kTTA_2 = 0
            self._kTTA_3 = 0
        elif self.TTA_channel == 2:  # this is T1 + T1 -> (TT)
            self._kTTA_1 = 0
            self._kTTA_2 = self.kTTA
            self._kTTA_3 = 0
        elif self.TTA_channel == 3:  # this is T1 + T1 -> S1
            self._kTTA_1 = 0
            self._kTTA_2 = 0
            self._kTTA_3 = self.kTTA
        else:
            raise ValueError('TTA channel must be either 1, 2 or 3')
        return
            
    def _initialise_simulation_tr(self):
        self._set_tta_rates()
        self._set_initial_condition(time_resolved=True)
        self._set_generation_rates(time_resolved=True)
        return

    def _unpack_simulation_tr(self, y):
        self.S1 = y[:, 1]
        self.TT = y[:, 2]
        self.T_T_total = np.sum(y[:, 3:12], axis=1)
        self.T1 = y[:, -1]
        self._wrap_simulation_results()
        return
    
    def _initialise_simulation_ss(self):
        self._set_tta_rates()
        self._set_initial_condition(time_resolved=False)
        self._set_generation_rates(time_resolved=False)
        return

    def _unpack_simulation_ss(self, y):
        self.S1 = y[1]
        self.TT = y[2]
        self.T_T_total = np.sum(y[3:12])
        self.T1 = y[-1]
        self._wrap_simulation_results()
        return
    
    def _wrap_simulation_results(self):
        self.simulation_results = dict(zip(self.states, [self.S1, self.TT, self.T_T_total, self.T1]))
        return
      

class Merrifield(KineticModelBase):
    """
    The standard Merrifield model.
    """
    def __init__(self):
        super().__init__()
        # metadata
        self.model_name = 'Merrifield'
        self.number_of_states = 11
        self.states = ['S1', 'TT_bright', 'TT_total', 'T_T_total', 'T1']
        self.rates = ['kGEN', 'kSF', 'k_SF', 'kDISS', 'KTTA', 'kRELAX', 'kSNR', 'kSSA', 'kTTNR', 'kTNR']
        # rates between excited states
        self.kSF = 20.0
        self.k_SF = 0.03
        self.kDISS = 0.067
        self.kTTA = 1e-18
        # spin relaxation (Bardeen addition - not in original Merrifield)
        self.kRELAX = 0
        # rates of decay
        self.kSNR = 0.1
        self.kSSA = 0
        self.kTTNR = 0.067
        self.kTNR = 1e-5
        # cslsq values
        self.cslsq = (1/9)*np.ones(9)

    def _rate_equations(self, y, t):
        GS, S1, TT_1, TT_2, TT_3, TT_4, TT_5, TT_6, TT_7, TT_8, TT_9, T1 = y
        dydt = np.zeros(self.number_of_states+1)
        # GS
        dydt[0] = -(self._kGENS+self._kGENT)*GS
        # S1
        dydt[1] = self._GS + self._kGENS*GS - (self.kSNR+self.kSF*np.sum(self.cslsq))*S1 -self.kSSA*S1*S1+ self.k_SF*(self.cslsq[0]*TT_1+self.cslsq[1]*TT_2+self.cslsq[2]*TT_3+self.cslsq[3]*TT_4+self.cslsq[4]*TT_5+self.cslsq[5]*TT_6+self.cslsq[6]*TT_7+self.cslsq[7]*TT_8+self.cslsq[8]*TT_9)
        # TT_1
        dydt[2] = self.kSF*self.cslsq[0]*S1 - (self.k_SF*self.cslsq[0]+self.kDISS+self.kTTNR+self.kRELAX)*TT_1 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_2+TT_3+TT_4+TT_5+TT_6+TT_7+TT_8+TT_9)
        # TT_2
        dydt[3] = self.kSF*self.cslsq[1]*S1 - (self.k_SF*self.cslsq[1]+self.kDISS+self.kTTNR+self.kRELAX)*TT_2 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_3+TT_4+TT_5+TT_6+TT_7+TT_8+TT_9)
        # TT_3
        dydt[4] = self.kSF*self.cslsq[2]*S1 - (self.k_SF*self.cslsq[2]+self.kDISS+self.kTTNR+self.kRELAX)*TT_3 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_4+TT_5+TT_6+TT_7+TT_8+TT_9)
        # TT_4
        dydt[5] = self.kSF*self.cslsq[3]*S1 - (self.k_SF*self.cslsq[3]+self.kDISS+self.kTTNR+self.kRELAX)*TT_4 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_5+TT_6+TT_7+TT_8+TT_9)
        # TT_5
        dydt[6] = self.kSF*self.cslsq[4]*S1 - (self.k_SF*self.cslsq[4]+self.kDISS+self.kTTNR+self.kRELAX)*TT_5 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_6+TT_7+TT_8+TT_9)
        # TT_6
        dydt[7] = self.kSF*self.cslsq[5]*S1 - (self.k_SF*self.cslsq[5]+self.kDISS+self.kTTNR+self.kRELAX)*TT_6 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_7+TT_8+TT_9)
        # TT_7
        dydt[8] = self.kSF*self.cslsq[6]*S1 - (self.k_SF*self.cslsq[6]+self.kDISS+self.kTTNR+self.kRELAX)*TT_7 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_6+TT_8+TT_9)
        # TT_8
        dydt[9] = self.kSF*self.cslsq[7]*S1 - (self.k_SF*self.cslsq[7]+self.kDISS+self.kTTNR+self.kRELAX)*TT_8 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_6+TT_7+TT_9)
        # TT_9
        dydt[10] = self.kSF*self.cslsq[8]*S1 - (self.k_SF*self.cslsq[8]+self.kDISS+self.kTTNR+self.kRELAX)*TT_9 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_6+TT_7+TT_8)
        # T1
        dydt[11] = self._GT + self._kGENT*GS + 2.0*self.kDISS*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_6+TT_7+TT_8+TT_9) - 2.0*self.kTTA*T1*T1 - self.kTNR*T1
        #
        return dydt

    def _unpack_simulation_tr(self, y):
        self.S1 = y[:, 1]
        self.TT_bright = self.cslsq[0]*y[:, 2] + self.cslsq[1]*y[:, 3] + self.cslsq[2]*y[:, 4] + self.cslsq[3]*y[:, 5] + self.cslsq[4]*y[:, 6] + self.cslsq[5]*y[:, 7] + self.cslsq[6]*y[:, 8] + self.cslsq[7]*y[:, 9] + self.cslsq[8]*y[:, 10]
        self.TT_total = np.sum(y[:, 2:11], axis=1)
        self.T_T_total = np.sum(y[:, 11:-1], axis=1)
        self.T1 = y[:, -1]
        self._wrap_simulation_results()
        return
        
    def _unpack_simulation_ss(self, y):
        self.S1 = y[1]
        self.TT_bright = self.cslsq[0]*y[2] + self.cslsq[1]*y[3] + self.cslsq[2]*y[4] + self.cslsq[3]*y[5] + self.cslsq[4]*y[6] + self.cslsq[5]*y[7] + self.cslsq[6]*y[8] + self.cslsq[7]*y[9] + self.cslsq[8]*y[10]
        self.TT_total = np.sum(y[2:11])
        self.T_T_total = np.sum(y[11:-1])
        self.T1 = y[-1]
        self._wrap_simulation_results()
        return
    
    def _wrap_simulation_results(self):
        self.simulation_results = dict(zip(self.states, [self.S1, self.TT_bright, self.TT_total, self.T_T_total, self.T1]))
        return
        

class Bardeen(KineticModelBase):
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
        self.rates = ['kGEN', 'kSF', 'k_SF', 'kHOP', 'k_HOP', 'kRELAX', 'kSNR', 'kSSA', 'kTTNR', 'kSPIN']
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
        self.kSSA =0
        self.kTTNR = 0.067
        self.kSPIN = 2.5e-4
        # cslsq values
        self.cslsq = (1/9)*np.ones(9)

    def _rate_equations(self, y, t):
        GS, S1, TT_1, TT_2, TT_3, TT_4, TT_5, TT_6, TT_7, TT_8, TT_9, T_T_1, T_T_2, T_T_3, T_T_4, T_T_5, T_T_6, T_T_7, T_T_8, T_T_9 = y
        dydt = np.zeros(self.number_of_states+1)
        # GS
        dydt[0] = -self._kGEN*GS
        # S1
        dydt[1] = self._GS + self._kGEN*GS - (self.kSNR+self.kSF*np.sum(self.cslsq))*S1 -self.kSSA*S1*S1 + self.k_SF*(self.cslsq[0]*TT_1+self.cslsq[1]*TT_2+self.cslsq[2]*TT_3+self.cslsq[3]*TT_4+self.cslsq[4]*TT_5+self.cslsq[5]*TT_6+self.cslsq[6]*TT_7+self.cslsq[7]*TT_8+self.cslsq[8]*TT_9)
        # TT_1
        dydt[2] = self.kSF*self.cslsq[0]*S1 - (self.k_SF+self.kTTNR)*self.cslsq[0]*TT_1 - self.kHOP*TT_1 + self.k_HOP*T_T_1
        # TT_2
        dydt[3] = self.kSF*self.cslsq[1]*S1 - (self.k_SF+self.kTTNR)*self.cslsq[1]*TT_2 - self.kHOP*TT_2 + self.k_HOP*T_T_2
        # TT_3
        dydt[4] = self.kSF*self.cslsq[2]*S1 - (self.k_SF+self.kTTNR)*self.cslsq[2]*TT_3 - self.kHOP*TT_3 + self.k_HOP*T_T_3
        # TT_4
        dydt[5] = self.kSF*self.cslsq[3]*S1 - (self.k_SF+self.kTTNR)*self.cslsq[3]*TT_4 - self.kHOP*TT_4 + self.k_HOP*T_T_4
        # TT_5
        dydt[6] = self.kSF*self.cslsq[4]*S1 - (self.k_SF+self.kTTNR)*self.cslsq[4]*TT_5 - self.kHOP*TT_5 + self.k_HOP*T_T_5
        # TT_6
        dydt[7] = self.kSF*self.cslsq[5]*S1 - (self.k_SF+self.kTTNR)*self.cslsq[5]*TT_6 - self.kHOP*TT_6 + self.k_HOP*T_T_6
        # TT_7
        dydt[8] = self.kSF*self.cslsq[6]*S1 - (self.k_SF+self.kTTNR)*self.cslsq[6]*TT_7 - self.kHOP*TT_7 + self.k_HOP*T_T_7
        # TT_8
        dydt[9] = self.kSF*self.cslsq[7]*S1 - (self.k_SF+self.kTTNR)*self.cslsq[7]*TT_8 - self.kHOP*TT_8 + self.k_HOP*T_T_8
        # TT_9
        dydt[10] = self.kSF*self.cslsq[8]*S1 - (self.k_SF+self.kTTNR)*self.cslsq[8]*TT_9 - self.kHOP*TT_9 + self.k_HOP*T_T_9
        # T_T_1
        dydt[11] = self.kHOP*TT_1 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_1 + (1/8)*self.kRELAX*(T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_2
        dydt[12] = self.kHOP*TT_2 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_2 + (1/8)*self.kRELAX*(T_T_1+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_3
        dydt[13] = self.kHOP*TT_3 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_3 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_4
        dydt[14] = self.kHOP*TT_4 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_4 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_5
        dydt[15] = self.kHOP*TT_5 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_5 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_6
        dydt[16] = self.kHOP*TT_6 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_6 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_7+T_T_8+T_T_9)
        # T_T_7
        dydt[17] = self.kHOP*TT_7 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_7 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_8+T_T_9)
        # T_T_8
        dydt[18] = self.kHOP*TT_8 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_8 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_9)
        # T_T_9
        dydt[19] = self.kHOP*TT_9 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_9 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8)
        #
        return dydt

    def _unpack_simulation_tr(self, y):
        self.GS = y[:, 0]
        self.S1 = y[:, 1]
        self.TT_bright = self.cslsq[0]*y[:, 2] + self.cslsq[1]*y[:, 3] + self.cslsq[2]*y[:, 4] + self.cslsq[3]*y[:, 5] + self.cslsq[4]*y[:, 6] + self.cslsq[5]*y[:, 7] + self.cslsq[6]*y[:, 8] + self.cslsq[7]*y[:, 9] + self.cslsq[8]*y[:, 10]
        self.TT_total = np.sum(y[:, 2:11], axis=1)
        self.T_T_total = np.sum(y[:, 11:], axis=1)
        self._wrap_simulation_results()
        return
    
    def _unpack_simulation_ss(self, y):
        self.GS = y[0]
        self.S1 = y[1]
        self.TT_bright = self.cslsq[0]*y[2] + self.cslsq[1]*y[3] + self.cslsq[2]*y[4] + self.cslsq[3]*y[5] + self.cslsq[4]*y[6] + self.cslsq[5]*y[7] + self.cslsq[6]*y[8] + self.cslsq[7]*y[9] + self.cslsq[8]*y[10]
        self.TT_total = np.sum(y[2:11])
        self.T_T_total = np.sum(y[11:])
        self._wrap_simulation_results()
        return
    
    def _wrap_simulation_results(self):
        self.simulation_results = dict(zip(self.states, [self.S1, self.TT_bright, self.TT_total, self.T_T_total]))
        return
