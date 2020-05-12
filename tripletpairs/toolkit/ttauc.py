"""Module for exploring the parameter space of TTA-UC acceptor systems."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def parameter_variation(m, sh, thing_to_vary, thing_to_divide_by, factors, num_points, draw_plot=True):
    """
    Explore the effect of varying different parameters on the upconversion yield of the acceptor.

    Parameters
    ----------
    m : tripletpairs.kineticmodelling.steadystatemodels.Merrifield or tripletpairs.kineticmodelling.steadystatemodels.MerrifieldExplicit1TT
        A pre-prepared instance of either :class:`tripletpairs.kineticmodelling.steadystatemodels.Merrifield` or :class:`tripletpairs.kineticmodelling.steadystatemodels.MerrifieldExplicit1TT`.
    sh : tripletpairs.spin.SpinHamiltonian
        A pre-prepared instance of :class:`tripletpairs.spin.SpinHamiltonian`.
    thing_to_vary : str
        The name of the rate constant to vary.
    thing_to_divide_by : str or None
        The name of the rate constant to normalise to, if desired.
    factors : 2-tuple of float
        Rate constant will be varied geometrically between its starting value divided by the first entry in factors and its starting value multiplied by the second entry in factors.
    num_points : int
        Number of rate constant values to sample.
    draw_plot : bool, optional
        Whether to draw a plot of the result. The default is True.

    Raises
    ------
    TypeError
        If the model given is invalid.
    ValueError
        If either of the given parameters is invalid.

    Returns
    -------
    rates : numpy.ndarray
        The rate constant values of thing_to_vary, note that these have not been divided by anything.
    ucy_actual : numpy.ndarray
        The upconversion yield as a function of rates.
    ucy_nospin : numpy.ndarray
        The upconversion yield as a function of rates, assuming no spin statistical effects.

    """
    if (m.model_name not in ['Merrifield', 'MerrifieldExplicit1TT']) or (m._time_resolved):
        raise TypeError('invalid model')
        
    m.initial_weighting = {'T1': 1}
    
    ucy_nospin = np.zeros(num_points)
    ucy_actual = np.zeros(num_points)
    
    if thing_to_vary not in m.rates:
        if thing_to_vary not in ['G', 'J']:
            raise ValueError('invalid thing_to_vary')
    if thing_to_divide_by is not None:
        if thing_to_divide_by not in m.rates:
            raise ValueError('invalid thing_to_divide_by')
        elif thing_to_vary in ['G', 'J']:
            raise ValueError('thing_to_divide_by must be None if thing_to_vary is G or J')
            
    if thing_to_vary == 'J':
        vars_object = sh
    else:
        vars_object = m
        
    vline = vars(vars_object)[thing_to_vary]
    
    rates = np.geomspace(vars(vars_object)[thing_to_vary]/factors[0], vars(vars_object)[thing_to_vary]*factors[1], num_points)
    for i, rate in enumerate(rates):
        
        vars(vars_object)[thing_to_vary] = rate
        
        sh.calculate_everything()
        m.cslsq = sh.cslsq
        m.simulate()
        ucy0 = 2*m.kSNR*m.S1/m.G
        
        m.cslsq = np.ones(9)/9
        m.simulate()
        ucy1 = 2*m.kSNR*m.S1/m.G
        
        ucy_nospin[i] = 100*ucy1
        ucy_actual[i] = 100*ucy0
        
    if draw_plot:
     
        rate_labels = {
            'kSF': 'Forwards SF Rate',
            'k_SF' : 'Backwards SF Rate',
            'kHOP' : 'Fowards Hop Rate',
            'k_HOP' : 'Backwards Hop Rate',
            'kHOP2' : 'Spin Loss Rate',
            'kDISS' : 'Spin Loss Rate',
            'kTTA' : 'TTA Rate',
            'kTTNR' : r'$^1$(TT) Decay Rate',
            'kTNR' : 'Triplet Decay Rate',
            'kSNR' : 'Singlet Decay Rate',
            'G' : 'Generation Rate',
            'J' : 'Exchange Energy'}
        
        if thing_to_divide_by is None:
            x = rates
            xlabel_text = rate_labels[thing_to_vary]
            if thing_to_vary == 'kTTA':
                xlabel_unit = r' (nm$^3$ns$^{-1}$)' 
            elif thing_to_vary == 'G':
                xlabel_unit = r' (nm$^{-3}$ns$^{-1}$)'
            elif thing_to_vary == 'J':
                xlabel_unit = r' ($\mu$eV)'
            else:
                xlabel_unit = r' (ns$^{-1}$)'
        else:
            x = rates/vars(vars_object)[thing_to_divide_by]
            vline /= vars(vars_object)[thing_to_divide_by]
            xlabel_text = rate_labels[thing_to_vary]+'/'+rate_labels[thing_to_divide_by]
            if thing_to_vary == 'kTTA':
                xlabel_unit = r' (nm$^3$)'
            elif thing_to_divide_by == 'kTTA':
                xlabel_unit = r' (nm$^{-3}$)'
            else:
                xlabel_unit = ''
                
        fig, ax1 = plt.subplots(figsize=(5, 4))
        
        ax1.semilogx(x, ucy_nospin, 'b--', label='loss')
        ax1.semilogx(x, ucy_actual, 'b-', label='loss')
        ax1.set_ylim([0, 100])
        ax1.set_xlabel(xlabel_text+xlabel_unit, fontsize=20)
        ax1.set_ylabel('Upconversion Yield (%)', fontsize=20, color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        gain =  100*(ucy_nospin-ucy_actual)/ucy_actual
        ax2.semilogx(x, gain, 'r:')
        ax2.set_ylim([0, 1.1*max(gain)])
        ax2.set_ylabel('Potential Gain (%)', fontsize=20, color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax1.xaxis.set_major_locator(mticker.LogLocator(numticks=12))
        ax1.xaxis.set_minor_locator(mticker.LogLocator(subs=np.linspace(0.1, 0.9, 9), numticks=12))
        ax1.set_xlim([min(x), max(x)])
        ax1.axvline(vline, color='k', linestyle=':', linewidth=1, alpha=0.5)
        
        for ax in [ax1, ax2]:
            ax.tick_params(axis='both', which='major', labelsize=20, width=1.4, length=6)
            ax.tick_params(axis='both', which='minor', labelsize=20, width=1.4, length=3)
            for axis in ['top','bottom','left','right']:
              ax.spines[axis].set_linewidth(1.4)
              
    return rates, ucy_actual, ucy_nospin
