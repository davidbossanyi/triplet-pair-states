import pandas as pd
from tripletpairs.toolkit import MolecularOrientation

longaxis_atom_labels = [1, 2]

atoms = pd.read_csv('mercury_file.csv', index_col=0, header=0)
mo = MolecularOrientation(atoms, longaxis_atom_labels)
mo.rotate_A2B()
mo.plot_3D(result=True, view=(10, 10))