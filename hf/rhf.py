
from time import perf_counter
import numpy as np
# Alternate eigenvalue computer
# from scipy.linalg import eigh

import ints

class RHF:
    """RHF class, instantiated with molecule, basis and optionally\
    charge and multiplicity."""
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

        H_core = nuc_attr + kinetic
        Lambda_S, L_S = np.linalg.eigh(ovlp)
        S_inv = L_S @ np.diag(1/np.sqrt(Lambda_S)) @ np.transpose(L_S)
        F_0 = np.transpose(S_inv) @ H_core @ S_inv
        epsilon_0, C_0 = np.linalg.eigh(F_0)
        C_0 = (S_inv @ C_0).real

        # Set initial C
        self.C = C_0
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

    def compute_E(self):
        """Computes the RHF energy. Returns (energy, MOs)"""
        start = perf_counter()
        # grab the integrals

        ovlp, E_nuc, kinetic, nuc_attr, eri = self.ints
        H_core = nuc_attr + kinetic

        # Orthogonalization matrix
        Lambda_S, L_S = np.linalg.eigh(ovlp)
        S_inv = L_S @ np.diag(1/np.sqrt(Lambda_S)) @ np.transpose(L_S)

        # Initial fock matrix and MO coefficients
        F_0A = H_core # Use the core guess
        F_0 = S_inv.T @ F_0A @ S_inv
        epsilon_0, C_0 = np.linalg.eigh(F_0)
        C_0 = (S_inv @ C_0).real # Back-transform

        # Initial density
        D_0 = np.einsum("mk,nk->mn",C_0[:,:self.occ],C_0[:,:self.occ],optimize=True)

        # Initial energy
        E_elec = np.trace(D_0 @ (F_0A + H_core))
        print("\nE_elec: %.7f\n"%E_elec)
        E_tot = E_elec + E_nuc

        #print("Initial orbital energies: %sEh"%np.array2string(epsilon_0,precision=3,max_line_width=150,suppress_small=True,separator=''))
        print("Starting SCF: %.3fms"%(1000*(perf_counter() - start)))
        print("\nIter  0: %.6f Eh"%E_tot)

        D = D_0
        delta_E = 1
        previous_E = E_tot
        iteration = 0
        while(delta_E > 1e-6):
            start = perf_counter()
            iteration += 1

            # Using PREVIOUS density: H = T + V + 2J - K
            JK = np.einsum("ls,mnls->mn",
                           D,
                           2*eri - eri.transpose((0,2,1,3)),
                           optimize=True)

            F_MO = JK + H_core

            F = S_inv.T @ F_MO @ S_inv          # Orthogonalize fock
            _, C_MO = np.linalg.eigh(F)         # Diagonalize fock
            C = (S_inv @ C_MO).real             # Back transform

            # Compute new density
            D = np.einsum("mk,nk->mn",C[:,:self.occ],C[:,:self.occ],optimize=True)

            E_elec = np.trace(D @ (F_MO + H_core))

            delta_E = abs(E_elec + E_nuc - E_tot)
            E_tot = E_elec + E_nuc # Previous iteration's Etot above

            # Show that the fock matrix is diagonal in MO basis
            F_ij = np.einsum("mj,ni,mn->ij",C,C,F_MO, optimize=True)
            F_ij = np.abs(F_ij-F_ij.T)
            if (F_ij.sum() > 1e-7):
                print("Fock matrix is not orthogonal in the MO basis!")

            print("Iter%3d: %.6f Eh  %.3fms  ΔE=%.6f"%(iteration, E_tot, 1000*(perf_counter()-start),delta_E))
            if (iteration > 200):
                exit("SCF did not converge")

        print("\nFinal RHF energy: %.9f"%E_tot)

        self.C = C
        self.HOMO = C.T[self.occ-1]
        self.LUMO = C.T[self.occ]

        #print(np.array2string(D_0, precision=6, suppress_small=True, max_line_width=200))
        return E_tot, C
