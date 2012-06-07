#!/usr/bin/env python
'''
lattice_propogation
===================

Script which uses the FCIQMC algorithm without annihilation to propogate
positive and negative psips using Hamiltonians defined in both Hartree product
and Slater determinant bases for two spinless fermions on a `4\\times4` lattice
with periodic boundary conditions.  The Hamiltonian operator is:

.. math::

    H = - \sum_{<ij>} t \left( c^\dagger_j c_i + c^\dagger_i c_j - u n_i n_j \\right).

This is an example where the evolution of an FCIQMC calculation is
step-by-step identical in both first- and second-quantized basis sets.

This is an independent implementation of work originally performed by Michael
Kolodrubetz and Bryan Clark (Princeton).
'''

import lattice_fci
import numpy

if __name__ == '__main__':

    N = 2
    L = 4
    sys = lattice_fci.LatticeModel(N, L)
    tau = 0.1

    print('# Fermions on a Lattice')
    print('# =====================')
    print('''#
# Use a simple lattice Hamiltonian as an illustration of when the FCIQMC sign
# problem is identical in both Hartree product and Slater determinant bases.
# 
# Number of fermions: %i
# Lattice dimensions: %i
#
# Propogation data format: N labels of the single-particle functions in
# a many-fermion basis function followed by the weight of the positive and
# negative particles on that basis function.
#
# No annihilation is performed.
#''' % (sys.nfermions, sys.L)
    )

    init_pos_basis = (lattice_fci.LatticeSite(0, 0, sys.L), lattice_fci.LatticeSite(0, 1, sys.L))
    init_neg_basis = (lattice_fci.LatticeSite(0, 1, sys.L), lattice_fci.LatticeSite(0, 0, sys.L))

    (hartree_products, determinants) = lattice_fci.init_lattice_basis(sys.nfermions, sys.L)

    hartree_hamil = lattice_fci.HartreeLatticeHamiltonian(sys, hartree_products, tau)
    (val, vec) = hartree_hamil.eigh()
    print('# Lowest eigenvalues in a Hartree product basis: %s.' % ', '.join('%f' % v for v in val[:10]))

    det_hamil = lattice_fci.DeterminantLatticeHamiltonian(sys, determinants, tau)
    (val, vec) = det_hamil.eigh()
    print('# Lowest eigenvalues in Slater determinant basis: %s.' % ', '.join('%f' %v for v in val[:5]))

    print("#")

    for (hamil, label) in ((hartree_hamil, 'Hartree product'), (det_hamil, 'Slater determinant')):
        print("# %s" % "Propogating Hamiltonian in %s basis" % (label))
        print("# %s" % ("-"*len("Propogating Hamiltonian in %s basis" % (label))))
        print("#")
        pos = numpy.zeros(len(hamil.basis))
        neg = numpy.zeros(len(hamil.basis))
        for (indx, bfn) in enumerate(hamil.basis):
            if bfn == init_pos_basis:
                pos[indx] = 1
            elif bfn == init_neg_basis:
                neg[indx] = 1
        t = 0
        for tfinal in (0, 1, 2, 8):
            while abs(t - tfinal) > 1.e-10:
                t += tau
                (pos, neg) = hamil.propogate(pos, neg)
            print("# %s" % "tau=%.2f" % (t))
            print("# %s" % ("^"*len("tau=%.2f" % (t))))
            print("#")
            lattice_fci.print_two_fermion_wfn(hamil.basis, pos, neg, L)
            print('\n\n\n')
