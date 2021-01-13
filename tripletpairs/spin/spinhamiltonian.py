import numpy as np


class SpinHamiltonian:
    r"""
    A class for calculating and using the spin Hamiltonian for (T..T) states.
    
    Attributes
    ----------
    mu_B : float
        Value of the Bohr magneton in eV. Default is 5.788e-5.
    g : float
        Value of the electron g-factor. Default is 2.002.
    J : float
        Value of the intertriplet exchange energy in eV.
    D, E: float
        Values of the intratriplet zero-field splitting parameters in eV.
    X : float
        Value of the intertriplet dipole-dipole coupling strength in eV.
    rAB : 3-tuple of float
        Centre-of-mass vector (x, y, z) between the two molecules comprising the triplet-pair. Must be given in the molecular coordinate system of molecule A.
    alpha, beta, gamma : float
        Euler angles :math:`\alpha, \beta, \gamma` that would rotate molecule A onto molecule B using the ZX'Z'' convention, in radians. Must be calculated in the molecular coordinate system of molecule A.
    theta, phi : float
        Spherical polar angles :math:`\theta, \phi` defining the orientation of the magnetic field in the molecular coordinate system of molecule A.
    B : float
        Value of the external magnetic field strength in Tesla.
    H : numpy.ndarray
        The computed Hamiltonian matrix (9x9).
    eigenvalues : numpy.ndarray
        1D array containing the computed eigenvalues (the energies of the 9 (T..T) states).
    eigenvectors : numpy.ndarray
        The computed eigenvectors (columns).
    csl : numpy.ndarray
        1D array containing the overlap factors between the 9 eigenstates and the singlet. Complex numbers in general.
    cslsq : numpy.ndarray
        1D array containing the squared overlap factors between the 9 eigenstates and the singlet. These are what is used in kinetic models.
    sum_ctlsq : numpy.ndarray
        1D array containing the total squared overlap factors between the 9 eigenstates and the 3 spin-1 triplet pair states.
    """
    
    def __init__(self):
        self._set_constants()
        self._initialise_parameters()
        self._set_singlet_state()
        self._set_triplet_states()
        
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
    
    def _set_triplet_states(self):
        self._triplet_state_x = (1/np.sqrt(2))*np.array([0, 0, 0, 0, 0, 1, 0, -1, 0])
        self._triplet_state_y = (1/np.sqrt(2))*np.array([0, 0, -1, 0, 0, 0, 1, 0, 0])
        self._triplet_state_z = (1/np.sqrt(2))*np.array([0, 1, 0, -1, 0, 0, 0, 0, 0])
        return
        
           
    def _rotation_matrix(self):
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
        Calculate the zero-field Hamiltonian for molecule A.
        
        Notes
        -----
        Depends on D and E only.

        Returns
        -------
        None.

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
        Calculate the zero-field Hamiltonian for a single molecule.
        
        Notes
        -----
        Depends on D and E only.
        
        This is a precursor for calculating the zero-field Hamiltonian of molecule B.

        Returns
        -------
        None.

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
        Calculate the zero-field Hamiltonian for molecule B.
        
        Notes
        -----
        Depends on alpha, beta and gamma.

        Returns
        -------
        None.

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
        Calculate the Zeeman Hamiltonian.
        
        Notes
        -----
        Depends on theta and phi only.

        Returns
        -------
        None.

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
        Calculate the zintertriplet dipole-dipole Hamiltonian.
        
        Notes
        -----
        Depends on rAB only.

        Returns
        -------
        None.

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
        Calculate the exchange Hamiltonian.
        
        Notes
        -----
        Independent of any parameters.

        Returns
        -------
        None.

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
        Hx = np.sin(self.theta)*np.cos(self.phi)
        Hy = np.sin(self.theta)*np.sin(self.phi)
        Hz = np.cos(self.theta)
        return Hx, Hy, Hz
        
    def calculate_hamiltonian(self):
        """
        Calculate the total spin Hamiltonian.
        
        Notes
        -----
        All the constituent parts must be calculated first.

        Returns
        -------
        None.

        """
        self.H = self.g*self.mu_B*self.B*self._H_Z \
                    + self._H_ZF_A + self._H_ZF_B \
                    + self.X*self._H_dd \
                    + self.J*self._H_ex
        return
        
    def calculate_eigenstates(self):
        """
        Calculate the eigenvalues and eigenstates.

        Returns
        -------
        None.

        """
        self.eigenvalues, self.eigenstates = np.linalg.eigh(self.H)
        return
        
    def calculate_cslsq(self):
        """
        Calculate the singlet overlaps.

        Returns
        -------
        None.

        """
        self.csl = np.matmul(self._singlet_state, self.eigenstates)
        self.cslsq = np.abs(self.csl)**2
        return
    
    def calculate_total_ctlsq(self):
        """
        Calculate the total of the triplet overlaps.

        Returns
        -------
        None.

        """
        self._ctl_x = np.matmul(self._triplet_state_x, self.eigenstates)
        self._ctlsq_x = np.abs(self._ctl_x)**2
        self._ctl_y = np.matmul(self._triplet_state_y, self.eigenstates)
        self._ctlsq_y = np.abs(self._ctl_y)**2
        self._ctl_z = np.matmul(self._triplet_state_z, self.eigenstates)
        self._ctlsq_z = np.abs(self._ctl_z)**2
        self.sum_ctlsq = self._ctlsq_x + self._ctlsq_y + self._ctlsq_z
        return
    
    def calculate_everything(self):
        """
        Calculate the Hamiltonian, eigenstates, eigenvalues and overlaps all at the same time.

        Returns
        -------
        None.

        """
        self.calculate_exchange_hamiltonian()
        self.calculate_zerofield_hamiltonian_single_molecule()
        self.calculate_zerofield_hamiltonian_molecule_A()
        self.calculate_zerofield_hamiltonian_molecule_B()
        self.calculate_dipoledipole_hamiltonian()
        self.calculate_zeeman_hamiltonian()
        self.calculate_hamiltonian()
        self.calculate_eigenstates()
        self.calculate_cslsq()
        return
