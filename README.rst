ueg_fci
=======

Full Configuration Interaction (FCI) method for the uniform electron gas.

This is a particularly simple implementation in python: little attempt is made
to conserve memory or CPU time.  Nevertheless, it is useful for small test
calculations, in particular for investigating ideas about the sign problem in
the Full Configuration Interaction Quantum Monte Carlo (FCIQMC) method
discussed by Spencer, Blunt and Foulkes (J. Chem. Phys. 136, 054110 (2012);
arXiv:1110.5479).

Note that no attempt is made to tackle finite size effects.

Requires python 2.6 or later and numpy.  All code is compatible with python 3.

``ueg_fci.py``
    Python module containing classes and functions for performing FCI
    calculations on the uniform electron gas using Slater determinants,
    permanents or Hartree products.  Also contains functions for transforming
    the Hamiltonian matrix into matrices related to the sign problem in FCIQMC.
``ueg_sign_problem.py``
    Example script using the ueg_fci module.

Documentation
-------------

Documentation can be found in the ``doc`` subdirectory and (mostly) in the
docstrings of the python source files.  Full HTML documentation can be viewed
at `readthedocs <http://ueg_fci.readthedocs.org>`_.

Author
------

James Spencer, Imperial College London.

License
-------

Modified BSD License; see LICENSE or ueg_fci.py for more details.
