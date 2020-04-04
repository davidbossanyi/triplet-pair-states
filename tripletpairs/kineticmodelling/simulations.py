from tripletpairs.spin import SpinHamiltonian

class KineticSimulation:
    
    def __init__(self, kinetic_model):
        self.kinetic_model = kinetic_model
        self.spin_hamiltonian = SpinHamiltonian()
        