import numpy as np


class SpinHamiltonian:
    r"""
    A class for calculating and using the spin Hamiltonian for (T-C) states.
    
    Attributes
    ----------
    mu_B : float
        Value of the Bohr magneton in eV. Default is 5.788e-5.
    g : float
        Value of the electron g-factor. Default is 2.002.
    D, E: float
        Values of the intratriplet zero-field splitting parameters in eV.
    theta, phi : float
        Spherical polar angles :math:`\theta, \phi` defining the orientation of the magnetic field in the molecular coordinate system.
    B : float
        Value of the external magnetic field strength in Tesla.
    H : numpy.ndarray
        The computed Hamiltonian matrix (6x6).
    eigenvalues : numpy.ndarray
        1D array containing the computed eigenvalues (the energies of the 6 (T-C) states).
    eigenvectors : numpy.ndarray
        The computed eigenvectors (columns).
    Dp, Dm : numpy.ndarray
        1D array containing the overlap factors between the 6 eigenstates and the plus/minus doublet. Complex numbers in general.
    Dpsq, Dmsq : numpy.ndarray
        1D array containing the squared overlap factors between the 6 eigenstates and the plus/minus doublet. These are what is used in kinetic models.
    Dsq : numpy.ndarray
        1D array containing the value of Dpsq + Dmsq
    """
    
    def __init__(self):
        self._set_constants()
        self._initialise_parameters()
        self._set_doublet_states()
        
    def _set_constants(self):
        self.mu_B = 5.788e-5  # Bohr magneton in eV/T
        self.g = 2.002        # electron gyromagnetic ratio
        return
        
    def _initialise_parameters(self):
        self.D = 1e-6         # D parameter in eV
        self.E = 3e-6         # E parameter in eV
        self.theta = 0        # anlge between B and z
        self.phi = 0          # angle defining B direction in xy plane
        self.B = 0            # magnetic field in Tesla
        return
    
    def _set_doublet_states(self):
        self._doublet_state_p = (1/np.sqrt(3))*np.array([1, 0, 1j, 0, 1, 0])
        self._doublet_state_m = (1/np.sqrt(3))*np.array([0, -1, 0, 1j, 0, 1])
        return
    
    def calculate_zerofield_hamiltonian(self):
        """
        Calculate the zero-field Hamiltonian.
        
        Notes
        -----
        Depends on D and E only.

        Returns
        -------
        None.

        """
        D3 = self.D/3
        E = self.E
        H_ZF = np.array([[D3-E, 0, 0, 0, 0, 0],
                         [0, D3-E, 0, 0, 0, 0],
                         [0, 0, D3+E, 0, 0, 0],
                         [0, 0, 0, D3+E, 0, 0],
                         [0, 0, 0, 0,-2*D3, 0],
                         [0, 0, 0, 0, 0,-2*D3]])
        self._H_ZF = H_ZF
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
        H_Z = np.array([[ Hz, Hx-1j*Hy,-2j*Hz, 0, 2j*Hy, 0],
                        [ Hx+1j*Hy,-Hz, 0,-2j*Hz, 0, 2j*Hy],
                        [ 2j*Hz, 0, Hz, Hx-1j*Hy,-2j*Hx, 0],
                        [ 0, 2j*Hz, Hx+1j*Hy,-Hz, 0,-2j*Hx],
                        [-2j*Hy, 0, 2j*Hx, 0, Hz, Hx-1j*Hy],
                        [ 0,-2j*Hy, 0, 2j*Hx, Hx+1j*Hy,-Hz]])
        self._H_Z = 0.5*H_Z
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
        self.H = self.g*self.mu_B*self.B*self._H_Z + self._H_ZF
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
        
    def calculate_Dsq(self):
        """
        Calculate the overlaps.

        Returns
        -------
        None.

        """
        self.Dp = np.matmul(self._doublet_state_p, self.eigenstates)
        self.Dpsq = np.abs(self.Dp)**2
        self.Dm = np.matmul(self._doublet_state_m, self.eigenstates)
        self.Dmsq = np.abs(self.Dm)**2
        self.Dsq = self.Dpsq + self.Dmsq
        return
    
    def calculate_everything(self):
        """
        Calculate the Hamiltonian, eigenstates, eigenvalues and overlaps all at the same time.

        Returns
        -------
        None.

        """
        self.calculate_zerofield_hamiltonian()
        self.calculate_zeeman_hamiltonian()
        self.calculate_hamiltonian()
        self.calculate_eigenstates()
        self.calculate_Dsq()
        return


if __name__ == '__main__':
    
    fQ = 0.5
    
    sh = SpinHamiltonian()
    sh.D = 6.45e-6
    sh.E = -6.45e-7
    sh.calculate_zerofield_hamiltonian()
    sh.theta = 0
    sh.phi = 0
    sh.calculate_zeeman_hamiltonian()
    
    Bs = np.linspace(0, 0.3, 100)
    Qs = np.zeros_like(Bs)
    for i, B in enumerate(Bs):
        sh.B = B
        sh.calculate_hamiltonian()
        sh.calculate_eigenstates()
        sh.calculate_Dsq()
        Qs[i] = np.sum((fQ*sh.Dsq)/((1-fQ)+(fQ*sh.Dsq)))
        
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(1000*Bs, Qs/Qs[0])
    
        