from tripletpairs.spin import SpinHamiltonian
from tripletpairs.kineticmodelling import models
from tripletpairs.kineticmodelling import KineticSimulation

sh = SpinHamiltonian()

m = models.Merrifield()

sim = KineticSimulation(m)