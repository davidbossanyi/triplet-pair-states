Molecular Orientations
======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
Introduction
------------

Computation of Euler angles and centre-of-mass vectors between the molecules of a triplet-pair can be done using the :class:`tripletpairs.toolkit.MolecularOrientation` class. This will require the following steps to be performed in `Mercury <https://www.ccdc.cam.ac.uk/solutions/csd-system/components/mercury/>`_ first:

1. Open the CIF file in Mercury
2. Enable packing checkbox
3. Adjust the packing to show only the two molecules of interest (calculate -> packing/slicing -> adjust a, b, c)
4. Bring up the atom list
5. Ensure that **only** Xorth, Yorth and Zorth fields are shown (customise button)
6. Click save and choose csv file type
7. Record the number of two atoms in molecule A that lie parallel to the long-axis (x) and have the same y-coordinate

The Euler angles and centre-of-mass vector can then be computed using :class:`tripletpairs.toolkit.MolecularOrientation`.

Example
-------

.. code-block:: python

   import pandas as pd
   from tripletpairs.toolkit import MolecularOrientation
   # atoms parallel to molecular x-axis
   longaxis_atom_labels = [1, 2]
   # load the csv file output from Mercury as a pandas DataFrame
   atoms = pd.read_csv('mercury_file.csv', index_col=0, header=0)
   # use the class
   mo = MolecularOrientation(atoms, longaxis_atom_labels)
   mo.rotate_A2B()
   mo.plot_3D(result=True, view=(10, 10))

API
---

.. autoclass:: tripletpairs.toolkit.MolecularOrientation
	:members: