toy_fci
=======

Full Configuration Interaction (FCI) method for simple quantum systems.
Currently the uniform electron gas and a spinless lattice model are
implemented.

This is a particularly naive implementation in python: little attempt is made
to conserve memory or CPU time.  Nevertheless, it is useful for small test
calculations, in particular for investigating ideas about the sign problem in
the Full Configuration Interaction Quantum Monte Carlo (FCIQMC) method
discussed by Spencer, Blunt and Foulkes (J. Chem. Phys. 136, 054110 (2012);
arXiv:1110.5479).

Note that no attempt is made to tackle finite size effects.

Requires python 2.6 or later and numpy.  All code is compatible with python 3.

``hamil.py``
    Python module containing a base Hamiltonian class and generic functions
    for performing finding excitations connecting pairs of many-particle basis
    functions.
``lattice_fci.py``
    Python module containing classes and functions for performing FCI
    calculations on spinless fermions hopping on a lattice with nearest
    neighbour interactions using Slater determinants or Hartree products.
``ueg_fci.py``
    Python module containing classes and functions for performing FCI
    calculations on the uniform electron gas using Slater determinants,
    permanents or Hartree products.
``lattice_propogation.py``
    Example script using the lattice_fci module.
``ueg_sign_problem.py``
    Example script using the ueg_fci module.

Documentation
-------------

Documentation can be found in the ``doc`` subdirectory and (mostly) in the
docstrings of the python source files.  Full HTML documentation can be viewed
at `readthedocs <http://toy_fci.readthedocs.org>`_.

Author
------

James Spencer, Imperial College London.

License
-------

Modified BSD License; see LICENSE for more details.

Acknowledgments
---------------

The lattice model was proposed by Michael Kolodrubetz and Bryan Clark
(Princeton), who showed that an FCIQMC calculation on this system in a Hartree
product basis cannot be distinguished from one in a Slater determinant basis.
