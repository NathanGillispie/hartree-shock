
from time import perf_counter
import numpy as np
import ints

class wavefunction:
    """Base class"""

    def __init__(self, molecule, basis, charge=0, multiplicity=1):
        self.molecule = molecule
        self.basis = basis
        self.charge = charge
        self.mult = multiplicity

        mints = ints.integrals(molecule, basis)

        self.nbf = mints.nbf()
        self.occ = self.occupied_orbitals()

        # Compute integrals
        start = perf_counter()

        ovlp = mints.overlap()
        E_nuc = ints.nuclear_repulsion(molecule)
        kinetic = mints.kinetic_energy()
        nuc_attr = mints.nuclear_attraction()
        eri = mints.electron_repulsion()
        self.ints = (ovlp, E_nuc, kinetic, nuc_attr, eri)

        print("Integrals computed: %.3fms"%(1000*(perf_counter() - start)))
        start = perf_counter()

        H_core = nuc_attr + kinetic

        # Orthogonalization matrix
        Lambda_S, L_S = np.linalg.eigh(ovlp)
        # \mathbf{S}^{-1/2} aka AO2MO
        S_inv = L_S @ np.diag(1/np.sqrt(Lambda_S)) @ np.transpose(L_S)

        # Initial fock matrix and MO coefficients using core guess
        # Core guess means F_0' = H_core
        F_0 = np.transpose(S_inv) @ H_core @ S_inv
        self.MO_energies, self.C_MO = np.linalg.eigh(F_0)
        C_0 = S_inv @ self.C_MO

        # Initial density
        self.D = np.einsum("mk,nk->mn",C_0[:,:self.occ],C_0[:,:self.occ],optimize=True)

        print("Initializing SCF took %.3fms"%(1000*(perf_counter() - start)))

        # Set initial C
        self.C = C_0
        self.S_inv = S_inv
        self.HOMO = self.C.T[self.occ-1]
        self.LUMO = self.C.T[self.occ]

    def occupied_orbitals(self):
        """Returns number of occupied oribitals"""
        Z_total = np.asarray([a[0] for a in self.molecule], int).sum()
        num_elec = Z_total - self.charge
        unpaired_elec = self.mult - 1
        # n_elec + unpaired pairs all electrons to form whole MOs
        if ((num_elec + unpaired_elec) % 2 != 0):
            exit("Incompatible charge and multiplicity")
        occ = (num_elec + unpaired_elec)//2

        if (occ > self.nbf):
            exit("More occupied MOs than basis functions. This should never happen. Good luck.")
        if (occ == self.nbf):
            exit("No LUMO orbital, choose a large basis set")
        return occ

        f.close()

