from tripletpairs.toolkit import ttauc
from tripletpairs.kineticmodelling import steadystatemodels
from tripletpairs.spin import SpinHamiltonian

# choose which parameter to explore   and what (if anything) to normalise by  
thing_to_vary = 'kTNR'
thing_to_divide_by = None

# set the parameter sampling options
factors = (100, 100)
num_points = 51

# setup the spin Hamiltonian
sh = SpinHamiltonian()
sh.D = 6.45e-6
sh.J = 0
sh.E = sh.D/3
sh.X = sh.D/1000
sh.alpha, sh.beta, sh.gamma = 1.570796, 1.060889, 1.570796
sh.rAB = (0.063132, 0, 0.998005)
sh.B, sh.theta, sh.phi = 0, 0, 0

# set up the rate model
m = steadystatemodels.MerrifieldExplicit1TT()
m.kSF = 100
m.k_SF = 180
m.kHOP = 15
m.k_HOP = 0.01
m.kHOP2 = 1e-4
m.kTTA = 5e-3
m.kTTNR = 0.0015
m.kTNR = 1e-5
m.kSNR = 0.06
m.G = 2.7e-8

# do the simulations
rates, ucy_actual, ucy_nospin = ttauc.parameter_variation(m, sh, thing_to_vary, thing_to_divide_by, factors, num_points)
    