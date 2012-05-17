#!/usr/bin/env python

import itertools
import numpy
import pprint

#--- Basis functions ---

class BasisFn:
    def __init__(self, i, j, k, L, spin):
        '''Create a k-point with wavevector 2*pi*(i,j,k)/L.'''
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
    return sum(bfn.k for bfn in basis_iterable)

#--- System ---

class UEG:
    def __init__(self, nel, nalpha, nbeta, rs):
        self.nel = nel
        self.nalpha = nalpha
        self.nbeta = nbeta
        self.rs = rs
        self.L = self.rs*((4*numpy.pi*self.nel)/3)**(1.0/3.0)
        self.Omega = self.L**3
    def coulomb_int(self, q):
        return 4*numpy.pi / (self.Omega * numpy.dot(q, q))

#--- Basis set ---

def init_basis(sys, cutoff, sym):

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
    basis_fns.sort()

    alpha_strings = tuple(comb for comb in itertools.combinations(basis_fns[::2], sys.nalpha))
    beta_strings = tuple(comb for comb in itertools.combinations(basis_fns[1::2], sys.nbeta))

    determinants = []
    for astring in alpha_strings:
        for bstring in beta_strings:
            if all(total_momentum(astring) + total_momentum(bstring) == sym):
                determinants.append(astring+bstring)
    determinants = tuple(determinants)

    # Now unfold determinants into all Hartree products in the Hilbert space.
    hartree_products = tuple(prod for det in determinants for prod in itertools.permutations(det))

    return (basis_fns, hartree_products, determinants)

#--- Hartree product matrix elements ---

def create_fq_hamiltonian(hartree_products):
    pass

#--- Determinant matrix elements ---

def slt_cnd0(sys, det):
    
    hmatel = 0
    for (indx, bi) in enumerate(det):
        hmatel += bi.kinetic
        for bj in det[indx+1:]:
            if bi.spin == bj.spin:
                hmatel -= sys.coulomb_int(bi.kp - bj.kp)
                pass

    return hmatel


def slt_cnd2(sys, det1, det2):

    # Get excitation.
    from_1 = []
    to_2 = []
    nperm = 0
    nfound = 0
    for (indx, basis) in enumerate(det1):
        if basis not in det2:
            from_1.append(basis)
            # Number of permutations required to move basis fn to the end.
            nperm += len(det1) - indx - 1 + nfound
            nfound += 1
    nfound = 0
    for (indx, basis) in enumerate(det2):
        if basis not in det1:
            to_2.append(basis)
            # Number of permutations required to move basis fn to the end.
            nperm += len(det2) - indx - 1 + nfound
            nfound += 1

    hmatel = 0
    if len(from_1) == 2:
        if all(total_momentum(from_1) == total_momentum(to_2)):
            if from_1[0].spin == to_2[0].spin and from_1[1].spin == to_2[1].spin:
                # Coulomb
                hmatel += sys.coulomb_int(from_1[0].kp - to_2[0].kp)
            if from_1[0].spin == to_2[1].spin and from_1[1].spin == to_2[0].spin:
                # Exchange
                hmatel -= sys.coulomb_int(from_1[0].kp - to_2[1].kp)

    if nperm%2 == 1:
        hmatel = -hmatel

    return hmatel

def create_sq_hamiltonian(sys, determinants):

    ndets = len(determinants)
    hamil = numpy.zeros([ndets, ndets])

    for i in range(len(determinants)):
        di = determinants[i]
        hamil[i][i] = slt_cnd0(sys, di)
        for j in range(i+1, len(determinants)):
            dj = determinants[j]
            hamil[i][j] = slt_cnd2(sys, di, dj)
            hamil[j][i] = hamil[i,j]

    return hamil

if __name__ == '__main__':

    nel = 4
    nalpha = 2
    nbeta = 2
    rs = 1

    cutoff = 1
    gamma = numpy.zeros(3)

    sys = UEG(nel, nalpha, nbeta, rs)
    (basis_fns, hartree_products, determinants) = init_basis(sys, cutoff, gamma)

    hamil = create_sq_hamiltonian(sys, determinants)
    eigval = numpy.linalg.eigvalsh(hamil)
    print(eigval[0])
