from tripletpairs.spin import SpinHamiltonian
from tripletpairs.kineticmodelling import models
from tripletpairs.kineticmodelling import KineticSimulation

sh = SpinHamiltonian()
sh.calculate_everything()

m = models.Merrifield()
m.simulate_steady_state()
m.simulate_time_resolved()

sim = KineticSimulation(m)