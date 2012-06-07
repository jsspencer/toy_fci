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

Requires python 2.6 or later and numpy.  All code is compatible with python 3.

.. toctree::
   :maxdepth: 1

   hamil
   lattice_fci
   ueg_fci
   lattice_propogation
   ueg_sign_problem
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
