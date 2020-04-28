import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class MolecularOrientation:
    """
    A class to work out Euler angles for a pair of molecules.
    
    Refer to the examples for usage guidelines.
    
    Parameters
    ----------
    atoms_AB : pandas.DataFrame
        Atom coordinates (cartesian), indexed by atom number. The first half should correspond to molecule A and the second half **in the same order** to molecule B.
    atoms_Apx : list of int
        Atom numbers of two atoms parallel to the long axis (x) of molecule A. They must have the **same** y-coordinate.
        
    Attributes
    ----------
    atoms : pandas.DataFrame
        Internal storage of the original atom coordinates.
    atoms_ApX : numpy.ndarray
        Coordinates of the two atoms parallel to x, with molecule A centred at the origin (3x2).
    molA, molB : numpy.ndarray
        Coordinates of the atoms in molecules A and B, in the molecular coordinate system of A.
    molA_transformed : numpy.ndarray
        Coordinates of the atoms in molecules A mapped to molecule B, in the molecular coordinate system of A.
    rAB : numpy.ndarray
        Centre-of-mass vector from A to B, in the molecular coordinate system of A.
    
    """
    
    def __init__(self, atoms_AB, atoms_Apx):
        if not isinstance(atoms_AB, pd.DataFrame):
            raise TypeError('atoms_AB must be a pandas dataframe')
        if not isinstance(atoms_Apx, list):
            raise TypeError('atoms_Apx must be a list')
        self.atoms = atoms_AB
        self.atoms_ApX = np.transpose(atoms_AB.loc[atoms_Apx, :].values)
        self._separate_atoms()
        self._centre_system()
        self._rotate_to_molecular_axes()
        self._move_both_to_origin()
        return
    
    def _separate_atoms(self):
        self.molA = np.transpose(self.atoms.iloc[0:(int(self.atoms.values.shape[0]/2)), :].values)
        self.molB = np.transpose(self.atoms.iloc[(int(self.atoms.values.shape[0]/2)):, :].values)
        return
    
    def _rotate_to_molecular_axes(self):
        print('\nfinding rotation to molecular axes\n')
        x0 = np.zeros(3)
        x = least_squares(lambda x: self._residuals_AtoMX(x), x0, method='lm', verbose=1).x
        self.molA = self._rotate(self.molA, x)
        self.molB = self._rotate(self.molB, x)
        print('\ndone')
        return
        
    def _residuals_AtoMX(self, x):
        atoms_ApX_transformed = self._rotate(self.atoms_ApX, x)
        residuals = np.array([
            atoms_ApX_transformed[2, 0],
            atoms_ApX_transformed[2, 1],
            atoms_ApX_transformed[1, 0]-atoms_ApX_transformed[1, 1]])
        return residuals
    
    @staticmethod
    def _calculate_centroid(points):
        return np.mean(points, axis=1)[:, np.newaxis]
    
    def _centre_system(self):
        molA_centroid = self._calculate_centroid(self.molA)
        self.molA = self.molA-molA_centroid
        self.molB = self.molB-molA_centroid
        self.atoms_ApX = self.atoms_ApX-molA_centroid
        return
    
    def _move_both_to_origin(self):
        molA_centroid = self._calculate_centroid(self.molA)
        molB_centroid = self._calculate_centroid(self.molB)
        self._molA_origin = self.molA-molA_centroid
        self._molB_origin = self.molB-molB_centroid
        self._rA2B = molB_centroid-molA_centroid
        self.rAB = self._rA2B/np.linalg.norm(self._rA2B)
        return
    
    def plot_3D(self, A=True, B=True, result=False, view=(30, 30)):
        """
        Create a 3D plot of the atom positions.

        Parameters
        ----------
        A : bool, optional
            Whether to plot molecule A (blue). Should be aligned such that x is the long axis, y is the short axis and z is perpendicular to the molecular plane. The default is True.
        B : bool, optional
            Whether to plot molecule B (red). The default is True.
        result : bool, optional
            Whether to plot the result of the transformation (black squares). Should overlap exactly with molecule B. The default is False.
        view : 2-tuple of float, optional
            Viewing angles for the 3D plot. The default is (30, 30).

        Returns
        -------
        None.

        """
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')
        if A:
            ax.scatter(self.molA[0, :], self.molA[1, :], self.molA[2, :], marker='o', color='b')
        if B:
            ax.scatter(self.molB[0, :], self.molB[1, :], self.molB[2, :], marker='*', color='r')
        if result:
            ax.scatter(self.molA_transformed[0, :], self.molA_transformed[1, :], self.molA_transformed[2, :], marker='s', color='k')
        self._hack_to_get_equal_axis_scales(ax)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_zlabel('z', fontsize=14)
        ax.view_init(*view)
        plt.show()
        return
    
    @staticmethod       
    def _hack_to_get_equal_axis_scales(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1]-x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1]-y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1]-z_limits[0])
        z_middle = np.mean(z_limits)
        plot_radius = 0.5*max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle-plot_radius, x_middle+plot_radius])
        ax.set_ylim3d([y_middle-plot_radius, y_middle+plot_radius])
        ax.set_zlim3d([z_middle-plot_radius, z_middle+plot_radius])
        return
    
    @staticmethod
    def _rotation_matrix(euler):
        alpha, beta, gamma = euler
        
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)
        sin_beta  = np.sin(beta)
        cos_beta  = np.cos(beta)
        sin_gamma = np.sin(gamma)
        cos_gamma = np.cos(gamma)
        
        R = np.array([[ cos_alpha*cos_gamma-sin_alpha*cos_beta*sin_gamma,  sin_alpha*cos_gamma+cos_alpha*cos_beta*sin_gamma, sin_beta*sin_gamma],
                      [-cos_alpha*sin_gamma-sin_alpha*cos_beta*cos_gamma, -sin_alpha*sin_gamma+cos_alpha*cos_beta*cos_gamma, sin_beta*cos_gamma],
                      [ sin_alpha*sin_beta                              , -cos_alpha*sin_beta                              , cos_beta        ]])
        return R

    def _rotate(self, a, euler):
        R = self._rotation_matrix(euler)
        a_transformed = np.matmul(R, a)
        return a_transformed
    
    def _residuals_AtoB(self, x):
        molA_origin_transformed = self._rotate(self._molA_origin, x)
        residuals = self._molB_origin-molA_origin_transformed
        return residuals.reshape(-1)
    
    @staticmethod
    def _modulo_0toN(value, N):
        if value < 0:
            while value < 0:
                value += N
        else:
            while value > N:
                value -= N
        return value
    
    def _generate_initial_guess(self):
        XpXT = np.matmul(self._molB_origin, np.transpose(self._molA_origin))
        XXT = np.matmul(self._molA_origin, np.transpose(self._molA_origin))
        R = np.matmul(XpXT, np.linalg.inv(XXT))
        if np.allclose(R, np.eye(3), atol=1e-3, rtol=1e-5):
            x0 = np.zeros(3)
        else:
            beta = np.arccos(R[2, 2])
            gamma = np.arccos(R[1, 2]/np.sin(beta))
            alpha = np.arcsin(R[2, 0]/np.sin(beta))
            x0 = np.array([alpha, beta, gamma])
        return x0
    
    def rotate_A2B(self):
        """
        Work out the Euler angles that would rotate A onto B.
        
        Notes
        -----
        The results will be printed to the console.

        Returns
        -------
        None.

        """
        print('\nfinding rotation from A to B\n')
        x0 = self._generate_initial_guess()
        x = least_squares(lambda x: self._residuals_AtoB(x), x0, method='lm', verbose=1).x
        alpha, beta, gamma = x
        self._molA_origin_transformed = self._rotate(self._molA_origin, x)
        self.molA_transformed = self._molA_origin_transformed+self._rA2B
        alpha = self._modulo_0toN(alpha, 2*np.pi)
        beta = self._modulo_0toN(beta, 2*np.pi)
        gamma = self._modulo_0toN(gamma, 2*np.pi)
        cost = np.sum(0.5*self._residuals_AtoB(x)**2)
        norm_cost = cost/(self.molA.shape[0]*self.molA.shape[1])
        print('\ncost = {0}\nnorm. cost = {1}\n'.format(cost, norm_cost))
        print('alpha = {0}\nbeta = {1}\ngamma = {2}'.format(alpha, beta, gamma))
        print('\nrAB = ({0}, {1}, {2})'.format(self.rAB[0, 0], self.rAB[1, 0], self.rAB[2, 0]))
        print('\ndone')
        return
