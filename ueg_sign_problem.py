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
    def print_eigv(eigv):
        print('# Lowest %i eigenvalues:\n#' % (len(eigv)))
        for i in range(len(eigv)):
            print('%3i %f' % (i+1, eigv[i]))
        print('#')

    print("# Constructing <%s'|H|%s>...\n#" % (2*(label,)))
    hamil = Hamil(sys, basis)
    print_title("<%s'|H|%s>" % (2*(label,)), '^')
    eigval = hamil.eigvalsh()
    print_eigv(eigval[:nprint])
    print_title("-|<%s'|H|%s>|, |%s'> /= |%s>" % (4*(label,)), '^')
    hamil.negabs_off_diagonal_elements()
    eigval = hamil.eigvalsh()
    print_eigv(eigval[:nprint])
    print_title("-|<%s'|H|%s>|" % (2*(label,)), '^')
    hamil.negabs_diagonal_elements()
    eigval = hamil.eigvalsh()
    print_eigv(eigval[:nprint])

def print_title(title, under='='):
    '''Print the underlined title.

:param string title: section title to print out
:param string under: single character used to underline the title
'''
    print('# %s\n# %s\n#' % (title, under*len(title)))

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

    print_title('FCIQMC sign problem in the UEG', '=')

    print_title('Basis set', '-')
    (basis_fns, hartree_products, determinants) = ueg_fci.init_ueg_basis(sys, cutoff, gamma)
    print('# %i spin-orbitals\n# %i determinants/permanents\n# %i Hartree products\n#' 
            % (len(basis_fns), len(determinants), len(hartree_products)))

    # Construct relevant matrices and diagonalise them.
    print_title('Slater determinant basis', '-')
    worker('D', sys, determinants, ueg_fci.DeterminantUEGHamiltonian, 10)
    # Permanent basis is identical to the determinant basis.
    print_title('Permanent basis', '-')
    worker('P', sys, determinants, ueg_fci.PermanentUEGHamiltonian, 10)
    # This is slow and uses lots of memory due to the sheer size of the Hartree product basis.
    #print_title('Hartree product basis', '-')
    #worker('h', sys, hartree_products, ueg_fci.HartreeUEGHamiltonian, 100)
