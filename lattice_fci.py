'''
lattice_fci
===========

Implement the spinless fermion lattice model defined by the Hamiltonian

.. math::

    H = - \sum_{<ij>} t \left( c^\dagger_j c_i + c^\dagger_i c_j - u n_i n_j \\right)

in both Hartree product and Slater determinant bases on a square lattice of
dimension `L \\times L`.

This is a simple example of when the sign problem in FCIQMC is identical in
both first- and second-quantized bases.

This is an implementation of the model originally proposed by Michael
Kolodrubetz and Bryan Clark (Princeton).
'''

# copyright (c) 2012 James Spencer.
# All rights reserved.
#
# Modified BSD License; see LICENSE for more details.

import hamil
import itertools
import numpy

#--- single-particle basis function ---

class LatticeSite:
    '''Basis function of a lattice site at location `(x,y)` in a `L \\times L` square lattice.

:param integer x: x coordinate of lattice site.
:param integer y: y coordinate of lattice site.
:param integer L: dimension of lattice cell.
'''
    def __init__(self, x, y, L):
        #: x coordinate of lattice site.
        self.x = x
        #: y coordinate of lattice site.
        self.y = y
        #: unique index of lattice site.
        self.loc = L*x + y
    def __lt__(self, other):
        return self.loc < other.loc
    def __eq__(self, other):
        return self.loc == other.loc
    def __repr__(self):
        return (self.x, self.y, self.loc).__repr__()

#--- System ---

class LatticeModel:
    '''Representation of the lattice model system defined by the Hamiltonian:

.. math::

    H = - \sum_{<ij>} t \left( c^\dagger_j c_i + c^\dagger_i c_j - u n_i n_j \\right)

:param integer nfermions: dimension of simulation cell
:param integer L: number of spinless fermions in the simulation cell
:param float t: hopping (kinetic) parameter in Hamiltonian operator
:param float u: Coulomb parameter in Hamiltonian operator
'''
    def __init__(self, nfermions, L, t=1, u=1):

        #: number of spinless fermions in the simulation cell
        self.nfermions = nfermions
        #: dimension of simulation cell
        self.L = L
        #: Hopping parameter
        self.t = t
        #: Coulomb parameter
        self.u = u

    def hopping_int(self, site1, site2):
        '''Calculate the hopping integral between two fermions.
        
:type site1: :class:`LatticeSite`
:param site1: lattice site, `s_1`, occupied by a fermion
:type site2: :class:`LatticeSite`
:param site2: lattice site, `s_2`, occupied by a fermion

:returns: `u \sum_{<ij>} \langle s_1 | u n_i n_j | s_2 \\rangle`
'''

        x = abs(site1.x - site2.x)
        y = abs(site1.y - site2.y)
        if ( (x == 1 or x == self.L - 1) and y == 0) or ( x == 0 and (y == 1 or y == self.L - 1)):
            return -self.t
        else:
            return 0

    def coulomb_int(self, site1, site2):
        '''Calculate the Coulomb integral between two fermions.
        
:type site1: :class:`LatticeSite`
:param site1: lattice site, `s_1`, occupied by a fermion
:type site2: :class:`LatticeSite`
:param site2: lattice site, `s_2`, occupied by a fermion

:returns: `-t \sum_{<ij>} \langle s_1 |  c^\dagger_j c_i + c^\dagger_i c_j | s_2 \\rangle`
'''

        x = abs(site1.x - site2.x)
        y = abs(site1.y - site2.y)
        if ( (x == 1 or x == self.L - 1) and y == 0) or ( x == 0 and (y == 1 or y == self.L - 1)):
            return self.u
        else:
            return 0

#--- Basis sets ---

def init_lattice_basis(nfermions, L):
    '''Construct many-fermion bases.

:param integer nfermions: number of fermions in a simulation cell.
:param integer L: dimension of 2D square simulation cell, where each lattice
    site contains a single-partle basis function.

:returns: (hartree_producs, determinants) where:

    hartree_products
        tuple containing all Hartree products.
    determinants
        tuple containing all Slater determinants.

Determinants and Hartree products are represented as tuples of
:class:`LatticeSite` objects.
'''

    # List of all sites in unit cell.
    # This is the set of single-particle basis functions.
    sites = tuple( LatticeSite(x, y, L) for x in range(L) for y in range(L) )

    # Hartree products are simply all permutations of the single-particle basis
    # functions.
    hartree_products = tuple( itertools.permutations(sites, nfermions) )

    # Similarly Slater determinants are simply all combinations of the
    # single-particle basis functions.
    determinants = tuple( itertools.combinations(sites, nfermions) )

    return (hartree_products, determinants)

#--- Hamiltonian in a Hartree product basis ---

class HartreeLatticeHamiltonian(hamil.Hamiltonian):
    '''Hamiltonian for the fermion lattice model in a Hartree product basis.

sys must be a :class:`LatticeModel` object and the underlying single-particle
basis functions must be :class:`LatticeSite` objects.

The Hartree product basis is the set of all possible permutations of fermions
in the single-particle basis set.
'''
    def mat_fn_diag(self, b):
        '''Calculate a diagonal Hamiltonian matrix element.

:type b: iterable of :class:`LatticeSite` objects
:param b: a Hartree product basis function, `|b\\rangle`

:rtype: float
:returns:  `\langle b|H|b \\rangle`
'''
        # H = - \sum_{<ij>} (c_i^\dagger c_j + c_j^\dagger c_i - n_i n_j)
        # No kinetic (hopping) integrals for diagonal matrix elements.
        # Only Coulomb terms if there are fermions on neighbouring sites.
        hmatel = 0
        for i in range(len(b)):
            for j in range(i+1, len(b)):
                hmatel += self.sys.coulomb_int(b[i], b[j])

        return hmatel

    def mat_fn_offdiag(self, bi, bj):
        '''Calculate an off-diagonal Hamiltonian matrix element.

:type bi: iterable of :class:`LatticeSite` objects
:param bi: a Hartree product basis function, `|b_i\\rangle`
:type bj: iterable of :class:`LatticeSite` objects
:param bj: a Hartree product basis function, `|b_j\\rangle`

:rtype: float
:returns: `\langle b_i|H|b_j \\rangle`.
'''
        # H = - \sum_{<ij>} (c_i^\dagger c_j + c_j^\dagger c_i - n_i n_j)
        # Coulomb operator is diagonal in this basis.
        # Kinetic term only if excitation involves moving a fermion from
        # a lattice site to a connected lattice site.

        (from_i, to_j) = hamil.hartree_excitation(bi, bj)

        if len(from_i) == 1:
            hmatel = self.sys.hopping_int(from_i[0], to_j[0])
        else:
            hmatel = 0

        return hmatel

#--- Hamiltonian in a Slater determinant basis ---

class DeterminantLatticeHamiltonian(hamil.Hamiltonian):
    '''Hamiltonian for the fermion lattice model in a Slater determinant basis.

sys must be a :class:`LatticeModel` object and the underlying single-particle
basis functions must be :class:`LatticeSite` objects.

The Slater determinant basis is the set of all possible combinations of
fermions in the single-particle basis set.
'''

    def mat_fn_diag(self, b):
        '''Calculate a diagonal Hamiltonian matrix element.

:type b: iterable of :class:`LatticeSite` objects
:param b: a Slater determinant basis function, `|b\\rangle`

:rtype: float
:returns:  `\langle b|H|b \\rangle`
'''
        # H = - \sum_{<ij>} (c_i^\dagger c_j + c_j^\dagger c_i - n_i n_j)
        # No kinetic (hopping) integrals for diagonal matrix elements.
        # Only Coulomb terms if there are fermions on neighbouring sites.
        # n.b. Due to normalisation and lack of exchange, it turns out this is
        # actually identical to the Hartree product terms...
        hmatel = 0
        for i in range(len(b)):
            for j in range(i+1, len(b)):
                hmatel += self.sys.coulomb_int(b[i], b[j])

        return hmatel

    def mat_fn_offdiag(self, bi, bj):
        '''Calculate an off-diagonal Hamiltonian matrix element.

:type bi: iterable of :class:`LatticeSite` objects
:param bi: a Slater determinant basis function, `|b_i\\rangle`
:type bj: iterable of :class:`LatticeSite` objects
:param bj: a Slater determinant basis function, `|b_j\\rangle`

:rtype: float
:returns: `\langle b_i|H|b_j \\rangle`.
'''

        # H = - \sum_{<ij>} (c_i^\dagger c_j + c_j^\dagger c_i - n_i n_j)
        # Coulomb operator is diagonal in this basis.
        # Kinetic term only if excitation involves moving a fermion from
        # a lattice site to a connected lattice site.

        (from_i, to_j, nperm) = hamil.determinant_excitation(bi, bj)
        
        if len(from_i) == 1:
            hmatel = self.sys.hopping_int(from_i[0], to_j[0])
        else:
            hmatel = 0

        if nperm % 2 == 1:
            hmatel = -hmatel

        return hmatel

#--- Convert a wavefunction into output suitable for a heatmap ---

def print_wfn(basis, pos, neg):
    '''Print out a stochastic wavefunction represented on a basis.

:type basis: iterable of iterable of :class:`LatticeSite` objects
:param basis: many-fermion basis
:type pos: 1D vector of length basis
:param pos: weight of positive psips on each basis function
:type neg: 1D vector of length basis
:param neg: weight of negative psips on each basis function
'''

    basis_coefs = list(zip(basis, pos, neg))
    basis_coefs.sort()
    (sorted_basis, sorted_pos, sorted_neg) = zip(*basis_coefs)

    irow = 0
    for i in range(len(sorted_basis)):
        # slightly weird syntax for python 2 and python 3 compatibility.
        if sorted_basis[i][0] != irow:
            irow = sorted_basis[i][0]
            print('')
        print('%s %s %s' % ' '.join('%i' % b.loc for b in sorted_basis[i]), sorted_pos[i], -sorted_neg[i])

def print_two_fermion_wfn(basis, pos, neg, L):
    '''Print out a stochastic wavefunction represented on a basis.

Assumes there are two fermions in the simulation cell.

:type basis: iterable of iterable of :class:`LatticeSite` objects
:param basis: many-fermion basis
:type pos: 1D vector of length basis
:param pos: weight of positive psips on each basis function
:type neg: 1D vector of length basis
:param neg: weight of negative psips on each basis function
:param integer L: dimension of lattice cell.
'''

    nbasis = len(basis)
    pos_wfn = numpy.zeros( (L**2, L**2) )
    neg_wfn = numpy.zeros( (L**2, L**2) )
    for i in range(nbasis):
        pos_wfn[basis[i][0].loc, basis[i][1].loc] = pos[i]
        neg_wfn[basis[i][0].loc, basis[i][1].loc] = neg[i]

    irow = 0
    for i in range(L**2):
        for j in range(L**2):
            print('%s %s %s %s' % (i, j, pos_wfn[i, j], -neg_wfn[i, j]))
        print('')
