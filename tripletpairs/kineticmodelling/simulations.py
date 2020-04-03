from triplet_pairs.spin import spinhamiltonian as sh

class KineticSimulation:
    
    def __init__(self, kinetic_model):
        self.kinetic_model = kinetic_model
        self.spin_hamiltonian = sh.SpinHamiltonian()
        