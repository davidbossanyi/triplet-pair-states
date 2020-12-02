import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq


def resample(x, y, new_x, smoothing=0):
    """
    Interpolate a dataset (x, y) onto a new set of points x with optional smoothing.

    Parameters
    ----------
    x : numpy.ndarray
        1D array of original x points.
    y : numpy.ndarray
        1D array of original y points.
    new_x : numpy.ndarray
        1D array of new x points.
    smoothing : float, optional
        Amount of smoothing to do. The default is 0.

    Returns
    -------
    new_y : numpy.ndarray
        The resampled y values.

    """
    y_spl = UnivariateSpline(x, y, s=smoothing)
    new_y = y_spl(new_x)
    return new_y


def convolve_irf(t, y, fwhm, shift_max_to_zero=False, normalise=False):
    """
    Perform a convolution with a gaussian IRF.

    Parameters
    ----------
    t : numpy.ndarray
        1D array containing the original time points.
    y : numpy.ndarray
        1D array to convolve with the IRF.
    fwhm : float
        Full width half maximum of the IRF in same time units as t.
    shift_max_to_zero : bool
        If True, the result will be shifted such that it is maximal at t = 0. The default is False.
    normalise : bool
        If True, the result will be normalised to its maximum value. The default is False.

    Returns
    -------
    t_irf : numpy.ndarray
        The new time points.
    y_irf : numpy.ndarray
        The convolved array.

    """
    w = fwhm/2.355
    linlog = 10*fwhm
    t_lin = np.linspace(0, 2*linlog, 1000)
    y_lin = resample(t, y, t_lin)
    t_full = np.hstack((-1*np.flip(t_lin[1:]), np.array(0), t_lin[1:]))
    y_full = np.hstack((np.zeros(len(t_lin)-1), y_lin))
    irf_full = np.exp(-(t_full**2)/(2*(w**2)))
    con = np.convolve(irf_full, y_full, mode='same')/np.sum(irf_full)
    t_irf = np.hstack((t_full[t_full < linlog], t[t >= linlog]))
    y_irf = np.hstack((con[t_full < linlog], y[t >= linlog]))
    if shift_max_to_zero:
        idx = np.argmax(y_irf)
        t_irf -= t_irf[idx]
    if normalise:
        y_irf /= max(y_irf)
    return t_irf, y_irf


def integrate_between(t, y, t1, t2):
        """
        Integrate the dynamics of y between times t1 and t2.

        Parameters
        ----------
        t : numpy.ndarray
            1D array containing the time points.
        y : numpy.ndarray
            1D array containing a simulated excited-state population.
        t1 : float
            Integrate **array** from this time to...
        t2 : float
            This time.

        Returns
        -------
        y_integrated : float
            The integrated value.

        """
        mask = ((t >= t1) & (t <= t2))
        y_integrated = np.trapz(y[mask], x=t[mask])/(t2-t1)
        return y_integrated
  
    
def normalise_at(t, y, tn):
        """
        Normalise y to time tn.

        Parameters
        ----------
        t : numpy.ndarray
            1D array containing the time points.
        y : numpy.ndarray
            1D array containing a simulated excited-state population.
        tn : float
            Time at which to normalise array.

        Returns
        -------
        y_norm : numpy.ndarray
            The normalised population.
        factor : float
            How much the original array was divided by.

        """
        idx = np.where((t-tn)**2 == min((t-tn)**2))[0][0]
        factor = y[idx]
        y_norm = y/factor
        return y_norm, factor
