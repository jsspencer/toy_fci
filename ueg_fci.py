#!/usr/bin/env python
'''Full Configuration Interaction (FCI) method for the uniform electron gas.

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

# copyright (c) 2012 James Spencer, Imperial College London.

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import itertools
import numpy
import pprint

#--- Basis functions ---

class BasisFn:
    '''Basis function with wavevector $2\pi(i,j,k)^{T}/L$ of the desired spin.

:param integer i, j, k: integer labels (quantum numbers) of the wavevector
:param float L: dimension of the cubic simulation cell of size $L\\times L \\times L$
:param integer spin: spin of the basis function (-1 for a down electron, +1 for an up electron)
'''
    def __init__(self, i, j, k, L, spin):
        self.k = numpy.array([x for x in (i,j,k)])
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

:returns: the total momentum, in units of $2\pi/L$, of the basis functions in basis_iterable
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
        '''Calculate the Coulomb integral $\langle  k k' | k+q k'-q  \\rangle$.

The Coulomb integral:

.. math::

    \langle k k' | k+q k'-q \\rangle = 4\pi/ (\Omega q^2)

where Omega is the volume of the simulation cell, is independent of  the
wavevectors $k$ and $k'$ and hence only the $q$ vector is required.

:type q: numpy.array
:param q: momentum transfer vector (in absolute units)
'''
        return 4*numpy.pi / (self.Omega * numpy.dot(q, q))

#--- Basis set ---

def init_basis(sys, cutoff, sym):
    '''Create single-particle and the many-particle bases.

:type sys: :class:`UEG`
:param sys: UEG system to be studied.

:param float cutoff: energy cutoff, in units of $(2\pi/L)^2$, defining the single-particle basis.  Only single-particle basis functions with a kinetic energy equal to or less than the cutoff are considered.

:type sym: numpy.array
:param sym: integer vector defining the wavevector, in units of $2\pi/L$, representing the desired symmetry.  Only Hartree products and determinants of this symmetry are returned.

:returns: (basis_fns, hartree_products, determinants)

# TODO

basis_fns: tuple of BasisFn objects, ie the single-particle basis set.
hartree_products: tuple containing Hartree products formed from basis_fns.  Warning: this can be large.
determinants: tuple containing Slater determinants formed from basis_fns.

A determinant and a Hartree product are both represented as tuples of BasisFn
objects.
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

#--- Construct Hamiltonian (base) ---

class UEGHamiltonian:
    '''Base Hamiltonian class for the UEG.

The relevant subclass which provides the appropriate matrix elements should be
used.

:type sys: :class:`UEG`
:param sys: UEG system to be studied
:type basis: iterable of iterables of :class:`ueg_fci.BasisFn`
:param basis: set of many-particle basis functions

.. note::

    This is a base class; the basis must be appropriate to the actual subclass
    used specific to the required underlying many-particle basis set.
'''
    def __init__(self, sys, basis):

        #: UEG system to be studied
        self.sys = sys
        #: set of many-particle basis functions
        self.basis = basis
        #: number of many-particle basis functions (i.e. length basis)
        self.nbasis = len(self.basis)
        #: Hamiltonian matrix in the basis set of the many-particle basis functions
        self.hamil = numpy.zeros([self.nbasis, self.nbasis])

        # Construct Hamiltonain
        for i in range(self.nbasis):
            bi = self.basis[i]
            self.hamil[i][i] = self.mat_fn0(bi)
            for j in range(i+1, self.nbasis):
                bj = self.basis[j]
                self.hamil[i][j] = self.mat_fn2(bi, bj)
                self.hamil[j][i] = self.hamil[i][j]

    def mat_fn0(self, b):
        '''Calculate diagonal matrix element, $\langle b|H|b \\rangle$.

.. warning::

    Virtual member.  Must be appropriately implemented in a subclass.

:type b: iterable of :class:`BasisFn` objects
:param b: a many-particle basis function.
'''

        err = 'Should not be calling the base matrix element functions'
        raise RuntimeError(err)

    def mat_fn2(self, b1, b2):
        '''Calculate an off-diagonal matrix element, $\langle b_1|H|b_2 \\rangle$.

.. warning::

    Virtual member.  Must be appropriately implemented in a subclass.

:type b1: iterable of :class:`BasisFn` objects
:param b1: a many-particle basis function
:type b2: iterable of :class:`BasisFn` objects
:param b2: a many-particle basis function
'''

        err = 'Should not be calling the base matrix element functions'
        raise RuntimeError(err)

    def eigvalsh(self):
        ''':returns: the eigenvalues of the Hamiltonian matrix.'''

        return numpy.linalg.eigvalsh(self.hamil)

    def negabs_off_diagonal_elements(self):
        '''Set off-diagonal elements of the Hamiltonian matrix to be negative.

This converts the Hamiltonian into the lesser sign-problem matrix discussed by
Spencer, Blunt and Foulkes.
'''

        for i in range(self.nbasis):
            for j in range(i+1,self.nbasis):
                self.hamil[i][j] = -abs(self.hamil[i][j])
                self.hamil[j][i] = -abs(self.hamil[j][i])

    def negabs_diagonal_elements(self):
        '''Set diagonal elements of the Hamiltonian matrix to be negative.

This, when called after negabs_offdiagonal_elements, converts the Hamiltonian
into the greater sign-problem matrix discussed by Spencer, Blunt and Foulkes.
'''

        for i in range(self.nbasis):
            self.hamil[i][i] = -abs(self.hamil[i][i])

#--- Hamiltonian in a Hartree product basis ---

class HartreeUEGHamiltonian(UEGHamiltonian):
    '''Hamiltonian class for the UEG in a Hartree product basis.

The Hartree product basis is the set of all possible permutations of electrons
in the single-particle basis set.  It is sufficient (and cheaper) to consider
one spin and momentum block of the Hamiltonian at a time.
'''
    def mat_fn0(self, p):
        '''Calculate a diagonal matrix element, $\langle p|H|p \\rangle$.

:type p: iterable of :class:`BasisFn` objects
:param p: a Hartree product basis function
'''
        # <p|H|p> = \sum_i <i|T|i>

        # Kinetic operator is diagonal in a plane-wave basis.

        # The Hartree energy from summing over coulomb integrals, <ij|ij>,
        # cancel exactly with background-background and electron-background
        # interactions.

        hmatel = 0
        for (indx, bi) in enumerate(p):
            hmatel += bi.kinetic
        return hmatel

    def mat_fn2(self, p1, p2):
        '''Calculate an off-diagonal matrix element, $\langle p_1|H|p_2 \\rangle$.

:type p1: iterable of :class:`BasisFn` objects
:param p1: a Hartree product basis function
:type p2: iterable of :class:`BasisFn` objects
:param p2: a many-particle basis function
'''
        # <p|H|p'> = <ij|U|ab>

        # Matrix element non-zero if exactly two spin-orbitals differ.
        # Order matters in Hartree products, so the differing spin-orbitals
        # must appear in the same place.

        # Kinetic operator is diagonal in a plane-wave basis, so only have
        # Coulomb integrals.

        # No exchange integrals in a Hartree product basis, of course.

        from_1 = []
        to_2 = []
        for i in range(len(p1)):
            if p1[i] != p2[i]:
                from_1.append(p1[i])
                to_2.append(p2[i])

        hmatel = 0
        if len(from_1) == 2:
            if from_1[0].spin == to_2[0].spin and from_1[1].spin == to_2[1].spin:
                # Coulomb
                hmatel = self.sys.coulomb_int(from_1[0].kp - to_2[0].kp)

        return hmatel

#--- Hamiltonian in a Slater determinant basis ---

class DeterminantUEGHamiltonian(UEGHamiltonian):
    '''Hamiltonian class for the UEG in a Slater determinant basis.

The Slater determinant basis is the set of all possible combinations of
electrons in the single-particle basis set.  It is sufficient (and cheaper) to
consider one spin and momentum block of the Hamiltonian at a time.
'''
    def mat_fn0(self, d):
        '''Calculate a diagonal matrix element, $\langle d|H|d \\rangle$.

:type d: iterable of :class:`BasisFn` objects
:param d: a Slater determinant basis function
'''
        # <D|H|D> = \sum_i <i|T|i> + \sum_{i<j} <ij|ij> - <ij|ji>

        # The Hartree energy from summing over coulomb integrals, <ij|ij>,
        # cancel exactly with background-background and electron-background
        # interactions.

        hmatel = 0
        for (indx, bi) in enumerate(d):
            hmatel += bi.kinetic
            for bj in det[indx+1:]:
                if bi.spin == bj.spin:
                    hmatel -= self.sys.coulomb_int(bi.kp - bj.kp)
                    pass

        return hmatel

    def mat_fn2(self, d1, d2):
        '''Calculate an off-diagonal matrix element, $\langle d_1|H|d_2 \\rangle$.

:type d1: iterable of :class:`BasisFn` objects
:param d1: a Slater determinant basis function
:type d2: iterable of :class:`BasisFn` objects
:param d2: a Slater determinant basis function

:returns: <d1|H|d2>
'''
        # <D|H|D'> = 1/2 <ij|ij> - <ij|ji>
        # if |D> and |D'> are related by a double excitation, assuming |D> and
        # |D'> are in maximum coincidence.

        # Get excitation.
        # Also work out the number of permutations required to line up the two
        # derminants.  We do this by counting the number of permutations
        # required to move the spin-orbitals to the 'end' of each determinant.
        from_1 = []
        to_2 = []
        nperm = 0
        nfound = 0
        for (indx, basis) in enumerate(d1):
            if basis not in d2:
                from_1.append(basis)
                # Number of permutations required to move basis fn to the end.
                # Have to take into account if we've already moved one orbital
                # to the end.
                nperm += len(d1) - indx - 1 + nfound
                nfound += 1
        nfound = 0
        # Ditto for second determinant.
        for (indx, basis) in enumerate(d2):
            if basis not in d1:
                to_2.append(basis)
                nperm += len(d2) - indx - 1 + nfound
                nfound += 1

        hmatel = 0
        if len(from_1) == 2:
            if all(total_momentum(from_1) == total_momentum(to_2)):
                if from_1[0].spin == to_2[0].spin and from_1[1].spin == to_2[1].spin:
                    # Coulomb
                    hmatel += self.sys.coulomb_int(from_1[0].kp - to_2[0].kp)
                if from_1[0].spin == to_2[1].spin and from_1[1].spin == to_2[0].spin:
                    # Exchange
                    hmatel -= self.sys.coulomb_int(from_1[0].kp - to_2[1].kp)

        if nperm%2 == 1:
            hmatel = -hmatel

        return hmatel

#--- Hamiltonian in a permanent basis ---

class PermanentUEGHamiltonian(UEGHamiltonian):
    '''Hamiltonian class for the UEG in a permanent basis.

The permanent basis is the set of all possible combinations of
electrons in the single-particle basis set, and hence is identical to the
Slater determinant basis.  It is sufficient (and cheaper) to consider one spin
and momentum block of the Hamiltonian at a time.
'''
    def mat_fn0(self, p):
        '''Calculate a diagonal matrix element, $\langle p|H|p \\rangle$.

:type p: iterable of :class:`BasisFn` objects
:param p: a permanent basis function
'''
        # <D|H|D> = \sum_i <i|T|i> + \sum_{i<j} <ij|ij> + <ij|ji>

        # The Hartree energy from summing over coulomb integrals, <ij|ij>,
        # cancel exactly with background-background and electron-background
        # interactions.

        hmatel = 0
        for (indx, bi) in enumerate(perm):
            hmatel += bi.kinetic
            for bj in perm[indx+1:]:
                if bi.spin == bj.spin:
                    hmatel += self.sys.coulomb_int(bi.kp - bj.kp)
                    pass

        return hmatel

    def mat_fn2(self, p1, p2):
        '''Calculate an off-diagonal matrix element, $\langle p_1|H|p_2 \\rangle$.

:type p1: iterable of :class:`BasisFn` objects
:param p1: a permanent basis function
:type p2: iterable of :class:`BasisFn` objects
:param p2: a permanent basis function
'''
        # <D|H|D'> = 1/2 <ij|ij> + <ij|ji>
        # if |D> and |D'> are related by a double excitation, assuming |D> and
        # |D'> are in maximum coincidence.

        # Get excitation.
        # No sign change associated with permuting spin orbitals in a permanent
        # in order to line them up, so don't need to count the permutations
        # required.
        from_1 = []
        to_2 = []
        for (indx, basis) in enumerate(p1):
            if basis not in p2:
                from_1.append(basis)
        for (indx, basis) in enumerate(p2):
            if basis not in p1:
                to_2.append(basis)

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
