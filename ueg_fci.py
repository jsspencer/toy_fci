#!/usr/bin/env python
'''
ueg_fci
=======

Full Configuration Interaction (FCI) method for the uniform electron gas.

This is a particularly simple implementation: little attempt is made to
conserve memory or CPU time.  Nevertheless, it is useful for small test
calculations, in particular for investigating ideas about the sign problem
in the Full Configuration Interaction Quantum Monte Carlo method discussed by
Spencer, Blunt and Foulkes (J. Chem. Phys. 136, 054110 (2012); arXiv:1110.5479).

FCI calculations can be performed in a Slater determinant, permanent or Hartree
product basis.

Note that no attempt is made to tackle finite size effects.

.. warning::

    The Hartree product basis set is much larger than the determinant/permanent
    basis set, even for tiny systems.  Thus the system sizes which can be
    tackled using the Hartree product basis are far more limited.
'''

# copyright (c) 2012 James Spencer.
# All rights reserved.
#
# Modified BSD License; see LICENSE for more details.

import hamil
import itertools
import numpy

#--- Basis functions ---

class BasisFn:
    '''Basis function with wavevector `2\pi(i,j,k)^{T}/L` of the desired spin.

:param integer i, j, k: integer labels (quantum numbers) of the wavevector
:param float L: dimension of the cubic simulation cell of size `L\\times L \\times L`
:param integer spin: spin of the basis function (-1 for a down electron, +1 for an up electron)
'''
    def __init__(self, i, j, k, L, spin):
        self.k = numpy.array([x for x in (i, j, k)])
        self.L = L
        self.kp = (self.k*2*numpy.pi)/L
        self.kinetic = numpy.dot(self.kp, self.kp)/2
        if not (spin == -1 or spin == 1):
            raise RuntimeError('spin not +1 or -1')
        self.spin = spin
    def __repr__(self):
        return (self.k, self.kinetic, self.spin).__repr__()
    def __lt__(self, other):
        return self.kinetic < other.kinetic

def total_momentum(basis_iterable):
    '''Calculate the total momentum of a many-particle basis function.

:type basis_iterable: iterable of :class:`BasisFn` objects
:param basis_iterable: many-particle basis function

:returns: the total momentum, in units of `2\pi/L`, of the basis functions in basis_iterable
'''
    return sum(bfn.k for bfn in basis_iterable)

#--- System ---

class UEG:
    '''Create a representation of a 3D uniform electron gas.

:param integer nel: number of electrons
:param integer nalpha: number of alpha (spin-up) electrons
:param integer nbeta: number of beta (spin-down) electrons
:param float rs: electronic density
'''
    def __init__(self, nel, nalpha, nbeta, rs):
        #: number of electrons
        self.nel = nel
        #: number of alpha (spin-up) electrons
        self.nalpha = nalpha
        #: number of beta (spin-down) electrons
        self.nbeta = nbeta
        #: electronic density
        self.rs = rs
        #: length of the cubic simulation cell containing nel electrons
        #: at the density of rs
        self.L = self.rs*((4*numpy.pi*self.nel)/3)**(1.0/3.0)
        #: volume of the cubic simulation cell containing nel electrons at the density
        #: of rs
        self.Omega = self.L**3
    def coulomb_int(self, q):
        '''Calculate the Coulomb integral `\langle  k \; k' | k+q \; k'-q  \\rangle`.

The Coulomb integral:

.. math::

    \langle k \; k' | k+q \; k'-q \\rangle = \\frac{4\pi}{\Omega q^2}

where `\Omega` is the volume of the simulation cell, is independent of  the
wavevectors `k` and `k'` and hence only the `q` vector is required.

:type q: numpy.array
:param q: momentum transfer vector (in absolute units)
'''
        return 4*numpy.pi / (self.Omega * numpy.dot(q, q))

#--- Basis set ---

def init_ueg_basis(sys, cutoff, sym):
    '''Create single-particle and the many-particle bases.

:type sys: :class:`UEG`
:param sys: UEG system to be studied.

:param float cutoff: energy cutoff, in units of `(2\pi/L)^2`, defining the single-particle basis.  Only single-particle basis functions with a kinetic energy equal to or less than the cutoff are considered.

:type sym: numpy.array
:param sym: integer vector defining the wavevector, in units of `2\pi/L`, representing the desired symmetry.  Only Hartree products and determinants of this symmetry are returned.

:returns: (basis_fns, hartree_products, determinants) where:

    basis_fns
        tuple of relevant :class:`BasisFn` objects, ie the single-particle basis set.
    hartree_products
        tuple containing all Hartree products formed from basis_fns.
    determinants
        tuple containing all Slater determinants formed from basis_fns.

Determinants and Hartree products are represented as tuples of :class:`BasisFn` objects.
'''

    # Single particle basis within the desired energy cutoff.
    cutoff = cutoff*(2*numpy.pi/sys.L)**2
    imax = int(numpy.ceil(numpy.sqrt(cutoff*2)))
    basis_fns = []
    for i in range(-imax, imax+1):
        for j in range(-imax, imax+1):
            for k in range(-imax, imax+1):
                bfn = BasisFn(i, j, k, sys.L, 1)
                if bfn.kinetic <= cutoff:
                    basis_fns.append(BasisFn(i, j, k, sys.L, 1))
                    basis_fns.append(BasisFn(i, j, k, sys.L, -1))
    # Sort in ascending order of kinetic energy.  Note that python's .sort()
    # (since 2.3) is guaranteed to be stable.
    basis_fns.sort()
    basis_fns = tuple(basis_fns)

    # All possible strings of up electons...
    alpha_strings = tuple(comb for comb in itertools.combinations(basis_fns[::2], sys.nalpha))
    # and all possible strings of down electons...
    beta_strings = tuple(comb for comb in itertools.combinations(basis_fns[1::2], sys.nbeta))

    # Combine all possible combinations of alpha and beta strings to give all
    # possible determinants.
    determinants = []
    for astring in alpha_strings:
        for bstring in beta_strings:
            if all(total_momentum(astring) + total_momentum(bstring) == sym):
                determinants.append(astring+bstring)
    determinants = tuple(determinants)

    # Now unfold determinants into all Hartree products in the Hilbert space.
    # Each Hartree product is only associated with a single determinant.  Each
    # determinant consists solely of a set of Hartree products related by
    # permutational symmetry.  Python's batteries make this very easy!
    hartree_products = tuple(prod for det in determinants for prod in itertools.permutations(det))

    return (basis_fns, hartree_products, determinants)

#--- Hamiltonian in a Hartree product basis ---

class HartreeUEGHamiltonian(hamil.Hamiltonian):
    '''Hamiltonian class for the UEG in a Hartree product basis.

sys must be a :class:`UEG` object and the single-particle basis functions must
be :class:`BasisFn` objects.

The Hartree product basis is the set of all possible permutations of electrons
in the single-particle basis set.  It is sufficient (and cheaper) to consider
one spin and momentum block of the Hamiltonian at a time.
'''
    def mat_fn_diag(self, p):
        '''Calculate a diagonal Hamiltonian matrix element.

:type p: iterable of :class:`BasisFn` objects
:param p: a Hartree product basis function, `|p_1\\rangle`

:rtype: float
:returns: `\langle p|H|p \\rangle`
'''
        # <p|H|p> = \sum_i <i|T|i>

        # Kinetic operator is diagonal in a plane-wave basis.

        # The Hartree energy from summing over coulomb integrals, <ij|ij>,
        # cancel exactly with background-background and electron-background
        # interactions.

        hmatel = 0
        for bi in p:
            hmatel += bi.kinetic
        return hmatel

    def mat_fn_offdiag(self, p1, p2):
        '''Calculate an off-diagonal matrix element.

:type p1: iterable of :class:`BasisFn` objects
:param p1: a Hartree product basis function, `|p_1\\rangle`
:type p2: iterable of :class:`BasisFn` objects
:param p2: a Hartree product basis function, `|p_2\\rangle`

:rtype: float
:returns:  `\langle p_1|H|p_2 \\rangle`
'''
        # <p|H|p'> = <ij|U|ab>

        # Matrix element non-zero if exactly two spin-orbitals differ.
        # Order matters in Hartree products, so the differing spin-orbitals
        # must appear in the same place.

        # Kinetic operator is diagonal in a plane-wave basis, so only have
        # Coulomb integrals.

        # No exchange integrals in a Hartree product basis, of course.

        (from_1, to_2) = hamil.hartree_excitation(p1, p2)

        hmatel = 0
        if len(from_1) == 2:
            if from_1[0].spin == to_2[0].spin and from_1[1].spin == to_2[1].spin:
                # Coulomb
                hmatel = self.sys.coulomb_int(from_1[0].kp - to_2[0].kp)

        return hmatel

#--- Hamiltonian in a Slater determinant basis ---

class DeterminantUEGHamiltonian(hamil.Hamiltonian):
    '''Hamiltonian class for the UEG in a Slater determinant basis.

sys must be a :class:`UEG` object and the single-particle basis functions must
be :class:`BasisFn` objects.

The Slater determinant basis is the set of all possible combinations of
electrons in the single-particle basis set.  It is sufficient (and cheaper) to
consider one spin and momentum block of the Hamiltonian at a time.
'''
    def mat_fn_diag(self, d):
        '''Calculate a diagonal Hamiltonian matrix element.

:type d: iterable of :class:`BasisFn` objects
:param d: a Slater determinant basis function, `|d\\rangle`

:rtype: float
:returns:  `\langle d|H|d \\rangle`
'''
        # <D|H|D> = \sum_i <i|T|i> + \sum_{i<j} <ij|ij> - <ij|ji>

        # The Hartree energy from summing over coulomb integrals, <ij|ij>,
        # cancel exactly with background-background and electron-background
        # interactions.

        hmatel = 0
        for (indx, bi) in enumerate(d):
            hmatel += bi.kinetic
            for bj in d[indx+1:]:
                if bi.spin == bj.spin:
                    hmatel -= self.sys.coulomb_int(bi.kp - bj.kp)

        return hmatel

    def mat_fn_offdiag(self, d1, d2):
        '''Calculate an off-diagonal Hamiltonian matrix element.

:type d1: iterable of :class:`BasisFn` objects
:param d1: a Slater determinant basis function, `|d_1\\rangle`
:type d2: iterable of :class:`BasisFn` objects
:param d2: a Slater determinant basis function, `|d_2\\rangle`

:rtype: float
:returns: `\langle d_1|H|d_2 \\rangle`.
'''
        # <D|H|D'> = 1/2 <ij|ij> - <ij|ji>
        # if |D> and |D'> are related by a double excitation, assuming |D> and
        # |D'> are in maximum coincidence.

        (from_1, to_2, nperm) = hamil.determinant_excitation(d1, d2)

        hmatel = 0
        if len(from_1) == 2:
            if all(total_momentum(from_1) == total_momentum(to_2)):
                if from_1[0].spin == to_2[0].spin and from_1[1].spin == to_2[1].spin:
                    # Coulomb
                    hmatel += self.sys.coulomb_int(from_1[0].kp - to_2[0].kp)
                if from_1[0].spin == to_2[1].spin and from_1[1].spin == to_2[0].spin:
                    # Exchange
                    hmatel -= self.sys.coulomb_int(from_1[0].kp - to_2[1].kp)

        if nperm % 2  == 1:
            hmatel = -hmatel

        return hmatel

#--- Hamiltonian in a permanent basis ---

class PermanentUEGHamiltonian(hamil.Hamiltonian):
    '''Hamiltonian class for the UEG in a permanent basis.

sys must be a :class:`UEG` object and the single-particle basis functions must
be :class:`BasisFn` objects.

The permanent basis is the set of all possible combinations of
electrons in the single-particle basis set, and hence is identical to the
Slater determinant basis.  It is sufficient (and cheaper) to consider one spin
and momentum block of the Hamiltonian at a time.
'''
    def mat_fn_diag(self, p):
        '''Calculate a diagonal Hamiltonian matrix element.

:type p: iterable of :class:`BasisFn` objects
:param p: a permanent basis function, `|p\\rangle`

:rtype: float
:returns: `\langle p|H|p \\rangle`.
'''
        # <D|H|D> = \sum_i <i|T|i> + \sum_{i<j} <ij|ij> + <ij|ji>

        # The Hartree energy from summing over coulomb integrals, <ij|ij>,
        # cancel exactly with background-background and electron-background
        # interactions.

        hmatel = 0
        for (indx, bi) in enumerate(p):
            hmatel += bi.kinetic
            for bj in p[indx+1:]:
                if bi.spin == bj.spin:
                    hmatel += self.sys.coulomb_int(bi.kp - bj.kp)

        return hmatel

    def mat_fn_offdiag(self, p1, p2):
        '''Calculate an off-diagonal Hamiltonian matrix element.

:type p1: iterable of :class:`BasisFn` objects
:param p1: a permanent basis function, `|p1\\rangle`
:type p2: iterable of :class:`BasisFn` objects
:param p2: a permanent basis function, `|p2\\rangle`

:rtype: float
:returns: `\langle p_1|H|p_2 \\rangle`.
'''
        # <P|H|P'> = 1/2 <ij|ij> + <ij|ji>
        # if |P> and |P'> are related by a double excitation.

        (from_1, to_2) = hamil.permanent_excitation(p1, p2)

        hmatel = 0
        if len(from_1) == 2:
            if all(total_momentum(from_1) == total_momentum(to_2)):
                if from_1[0].spin == to_2[0].spin and from_1[1].spin == to_2[1].spin:
                    # Coulomb
                    hmatel += self.sys.coulomb_int(from_1[0].kp - to_2[0].kp)
                if from_1[0].spin == to_2[1].spin and from_1[1].spin == to_2[0].spin:
                    # Exchange
                    hmatel += self.sys.coulomb_int(from_1[0].kp - to_2[1].kp)
        return hmatel
