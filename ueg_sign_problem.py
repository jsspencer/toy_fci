#!/usr/bin/env python
'''
ueg_sign_problem
================

Script to investigate the sign problem in a UEG system in various bases.

Execute to calculate and print out the lowest eigenvalues of Hamiltonian
matrices (and those related to the FCIQMC sign problem) for a small UEG
system using the ueg_fci module.
'''

import numpy
import ueg_fci

def worker(label, sys, basis, Hamil, nprint):
    '''Helper function to construct and diagonalise matrices related to the sign problem.

:param string label: label of the many-particle basis function used
:type sys: :class:`ueg_fci.UEG`
:param sys: desired UEG system; passed to Hamil
:type basis: iterable of iterables of :class:`ueg_fci.BasisFn`
:param basis: set of many-particle basis functions
:type Hamil: :class:`ueg_fci.UEGHamiltonian` subclass
:param Hamil: appropriate Hamiltonian for the basis provided
:param integer nprint: number of eigenvalues to print out
'''
    print("Constructing <%s'|H|%s>..." % (2*(label,)))
    hamil = Hamil(sys, basis)
    print("Diagonalising <%s'|H|%s>..." % (2*(label,)))
    eigval = hamil.eigvalsh()
    print(eigval[:nprint])
    print("Diagonalising -|<%s'|H|%s>|, |%s'> /= |%s>..." % (4*(label,)))
    hamil.negabs_off_diagonal_elements()
    eigval = hamil.eigvalsh()
    print(eigval[:nprint])
    print("Diagonalising -|<%s'|H|%s>|..." % (2*(label,)))
    hamil.negabs_diagonal_elements()
    eigval = hamil.eigvalsh()
    print(eigval[:nprint])

if __name__ == '__main__':

    # 4 electron UEG system at r_s=1 a.u.
    nel = 4
    nalpha = 2
    nbeta = 2
    rs = 1

    # Cutoff for the single-particle basis set.
    cutoff = 1.0
    # Symmetry of the many-particle wavefunction: consider gamma-point only.
    gamma = numpy.zeros(3)

    sys = ueg_fci.UEG(nel, nalpha, nbeta, rs)

    title = 'FCIQMC sign problem in the UEG'
    print(title)
    print('='*len(title))
    print('')

    print('Constructing the basis...')
    (basis_fns, hartree_products, determinants) = ueg_fci.init_ueg_basis(sys, cutoff, gamma)
    print('Basis set size: %i spin-orbitals, %i determinants/permanents, ' '%i Hartree products.' 
            % (len(basis_fns), len(determinants), len(hartree_products)))

    # Construct relevant matrices and diagonalise them.
    worker('D', sys, determinants, ueg_fci.DeterminantUEGHamiltonian, 10)
    # Permanent basis is identical to the determinant basis.
    worker('P', sys, determinants, ueg_fci.PermanentUEGHamiltonian, 10)
    # This is slow and uses lots of memory due to the sheer size of the Hartree product basis.
    worker('h', sys, hartree_products, ueg_fci.HartreeUEGHamiltonian, 100)
