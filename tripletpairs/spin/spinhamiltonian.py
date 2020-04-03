"""
Contains the class SpinHamiltonian. This class can be used to calculate the
overlaps of the triplet pair wavefunctions with the singlet triplet pair.

The Hamiltonian is constructed following the procedure described by Tapping
and Huang (https://doi.org/10.1021/acs.jpcc.6b04934).
"""

import numpy as np


class SpinHamiltonian:
    """
    ABOUT
    ---------------------------------------------------------------------------
    A class containing methods for calculating and using the total spin
    Hamiltonian for triplet-pair states (T..T).
    
    The Hamiltonian is the sum of the Zeeman, zero-field and dipole-dipole 
    terms.
    
    USAGE
    ---------------------------------------------------------------------------
    The D, E, and dipole-dipole interaction parameters can be adjusted as
    required, for example:
    
    >>> sh = SpinHamiltonian()
    >>> sh.D = 5.1e-6
    >>> sh.E = 1.6e-7
    >>> sh.X = 1.1e-9
    >>> sh.J = 0
    >>> sh.rAB = (0.7636, -0.4460, 0.4669)
    >>> sh.alpha, sh.beta, sh.gamma = 0, 0, 0
    >>> sh.theta, sh.phi = 0, 0
    >>> sh.B = 0
    
    The individual parts of the hamiltonian must be calculated separately:
    
    >>> sh.calculate_exchange_hamiltonian()
    >>> sh.calculate_zerofield_hamiltonian_single_molecule()
    >>> sh.calculate_zerofield_hamiltonian_molecule_A()
    >>> sh.calculate_dipoledipole_hamiltonian()
    >>> sh.calculate_zerofield_hamiltonian_molecule_B()
    >>> sh.calculate_zeeman_hamiltonian()
    
    >>> sh.calculate_hamiltonian()
    
    Then the quantities of interest can be calculated:
    
    >>> sh.calculate_TTl_states()
    >>> sh.calculate_cslsq()
    
    And accessed as attributes:
        
    >>> cslsq = sh.cslsq
    >>> eigenvalues = sh.TTl_eigenvalues
    >>> eigenstates = sh.TTl_eigenstates
    
    DETAILS
    ---------------------------------------------------------------------------
    uses constants:
        hbar = 1
    uses units:
        B:      tesla
        angles: radians
        energy: electronvolt 
    uses zero-field basis states:
        (x,x) (x,y) (x,z) (y,x) (y,y) (y,z) (z,x) (z,y) (z,z)
    uses Euler angle convention:
        ZX'Z''
    """
    def __init__(self):
        self._set_constants()
        self._initialise_parameters()
        self._set_singlet_state()
        
    def _set_constants(self):
        self.mu_B = 5.788e-5  # Bohr magneton in eV/T
        self.g = 2.002        # electron gyromagnetic ratio
        return
        
    def _initialise_parameters(self):
        self.D = 1e-6         # D parameter in eV
        self.E = 3e-6         # E parameter in eV
        self.X = 6e-8         # intertriplet dipole-dipole interaction in eV
        self.J = 0            # intertriplet exchange interaction in eV
        self.rAB = (0, 0, 1)  # unit vector from COM of A to COM of B in A coordinates
        self.alpha = 0        # euler rotation about z
        self.beta = 0         # euler rotation about x'
        self.gamma = 0        # euler rotation about z''
        self.theta = 0        # anlge between B and z
        self.phi = 0          # angle defining B direction in xy plane
        self.B = 0            # magnetic field in Tesla
        return
    
    def _set_singlet_state(self):
        self._singlet_state = (1/np.sqrt(3))*np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        return
        
           
    def _rotation_matrix(self):
        """
        Computes the 3x3 rotation matrix using the Euler angles that would
        rotate molecule A onto molecule B according to the zx'z'' convention.
        """
        sin_alpha = np.sin(self.alpha)
        cos_alpha = np.cos(self.alpha)
        sin_beta  = np.sin(self.beta)
        cos_beta  = np.cos(self.beta)
        sin_gamma = np.sin(self.gamma)
        cos_gamma = np.cos(self.gamma)
        
        R = np.array([[ cos_alpha*cos_gamma-sin_alpha*cos_beta*sin_gamma,  sin_alpha*cos_gamma+cos_alpha*cos_beta*sin_gamma, sin_beta*sin_gamma],
                      [-cos_alpha*sin_gamma-sin_alpha*cos_beta*cos_gamma, -sin_alpha*sin_gamma+cos_alpha*cos_beta*cos_gamma, sin_beta*cos_gamma],
                      [ sin_alpha*sin_beta                              , -cos_alpha*sin_beta                              , cos_beta        ]])
        return R
    
    def calculate_zerofield_hamiltonian_molecule_A(self):
        """
        Depends on E, D
        """
        D3 = self.D/3
        E = self.E
        H_ZF_A = np.array([[D3-E, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, D3-E, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, D3-E, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, D3+E, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, D3+E, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, D3+E, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0,-2*D3, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0,-2*D3, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0,-2*D3]])
        self._H_ZF_A = H_ZF_A
        return
    
    def calculate_zerofield_hamiltonian_single_molecule(self):
        """
        Depends on E, D
        """
        D3 = self.D/3
        E = self.E
        H_ZF_SM = np.array([[D3-E, 0, 0],
                            [0, D3+E, 0],
                            [0, 0,-2*D3]])
        self._H_ZF_SM = H_ZF_SM
        return
    
    def calculate_zerofield_hamiltonian_molecule_B(self):
        """
        Depends on E, D, alpha, beta, gamma
        """
        R = self._rotation_matrix()
        H_ZF_SM_B = np.matmul(np.transpose(R), np.matmul(self._H_ZF_SM, R))
        H_ZF_B = np.zeros((9, 9))
        H_ZF_B[0:3, 0:3] = H_ZF_SM_B
        H_ZF_B[3:6, 3:6] = H_ZF_SM_B
        H_ZF_B[6:9, 6:9] = H_ZF_SM_B
        self._H_ZF_B = H_ZF_B
        return
    
    def calculate_zeeman_hamiltonian(self):
        """
        Depends on theta, phi, later multiplied by g.muB.B
        """
        Hx, Hy, Hz = self._calculate_projections()
        H_Z = np.array([[  0,-Hz, Hy,-Hz,  0,  0, Hy,  0,  0],
                        [ Hz,  0,-Hx,  0,-Hz,  0,  0, Hy,  0],
                        [-Hy, Hx,  0,  0,  0,-Hz,  0,  0, Hy],
                        [ Hz,  0,  0,  0,-Hz, Hy,-Hx,  0,  0],
                        [  0, Hz,  0, Hz,  0,-Hx,  0,-Hx,  0],
                        [  0,  0, Hz,-Hy, Hx,  0,  0,  0,-Hx],
                        [-Hy,  0,  0, Hx,  0,  0,  0,-Hz, Hy],
                        [  0,-Hy,  0,  0, Hx,  0, Hz,  0,-Hx],
                        [  0,  0,-Hy,  0,  0, Hx,-Hy, Hx,  0]])
        self._H_Z = 1j*H_Z
        return
        
    def calculate_dipoledipole_hamiltonian(self):
        """
        Depends on rAB, later multiplied by X
        """
        u, v, w = self.rAB
        H_dd = np.array([[      0,       0,       0,       0, 1-3*w*w,   3*v*w,       0,   3*v*w, 1-3*v*v],
                         [      0,       0,       0,-1+3*w*w,       0,  -3*u*w,  -3*v*w,       0,   3*u*v],
                         [      0,       0,       0,  -3*v*w,   3*u*w,       0,-1+3*v*v,  -3*u*v,       0],
                         [      0,-1+3*w*w,  -3*v*w,       0,       0,       0,       0,  -3*u*w,   3*u*v],
                         [1-3*w*w,       0,   3*u*w,       0,       0,       0,   3*u*w,       0, 1-3*u*u],
                         [  3*v*w,  -3*u*w,       0,       0,       0,       0,  -3*u*v,-1+3*u*u,       0],
                         [      0,  -3*v*w,-1+3*v*v,       0,   3*u*w,  -3*u*v,       0,       0,       0],
                         [  3*v*w,       0,  -3*u*v,  -3*u*w,       0,-1+3*u*u,       0,       0,       0],
                         [1-3*v*v,   3*u*v,       0,   3*u*v, 1-3*u*u,       0,       0,       0,       0]])
        self._H_dd = -1*H_dd
        return
    
    def calculate_exchange_hamiltonian(self):
        """
        Depends on nothing, later multiplied by J
        """
        H_ex = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0,-1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0,-1, 0, 0],
                         [0,-1, 0, 0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0,-1, 0],
                         [0, 0,-1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0,-1, 0, 0, 0],
                         [1, 0, 0, 0, 1, 0, 0, 0, 0]])
        self._H_ex = H_ex
        return
        
    def _calculate_projections(self):
        """
        Projections of the B unit vector onto the x,y,z axis of molecule A
        """
        Hx = np.sin(self.theta)*np.cos(self.phi)
        Hy = np.sin(self.theta)*np.sin(self.phi)
        Hz = np.cos(self.theta)
        return Hx, Hy, Hz
        
    def calculate_hamiltonian(self):
        """
        Depends on everything
        """
        self.H = self.g*self.mu_B*self.B*self._H_Z \
                    + self._H_ZF_A + self._H_ZF_B \
                    + self.X*self._H_dd \
                    + self.J*self._H_ex
        return
        
    def calculate_eigenstates(self):
        """
        Depends on everything
        """
        self.eigenvalues, self.eigenstates = np.linalg.eigh(self.H)
        return
        
    def calculate_cslsq(self):
        """
        Depends on everything
        """
        self.csl = np.matmul(self._singlet_state, self.eigenstates)
        self.cslsq = np.abs(self.csl)**2
        return
