
from time import perf_counter
import numpy as np
# Alternate eigenvalue computer
# from scipy.linalg import eigh

import ints
from wavefunction import wavefunction
from utils import SCFConvergeError
import constants

class RHF(wavefunction):
    """RHF class, instantiated with molecule, basis and optionally\
    charge and multiplicity."""

    def compute_E(self, debug=False):
        """Computes the RHF energy. Returns (energy, MOs). Has debug flag"""
        start = perf_counter()

        # grab the integrals
        ovlp, E_nuc, kinetic, nuc_attr, eri = self.ints
        H_core = nuc_attr + kinetic

        # Initial energy
        E_elec = np.trace(self.D @ (H_core + H_core))
        E_tot = E_elec + E_nuc
        
        if (debug):
            print("Initial orbital energies: %sEh"%np.array2string(epsilon_0,precision=3,max_line_width=150,suppress_small=True,separator=''))
            print("\nIter  0: %.6f Eh"%E_tot)

        delta_E = 1
        previous_E = E_tot
        iteration = 0
        start_SCF = perf_counter()
        while(delta_E > 1e-8):
            iteration += 1

            # Using PREVIOUS density: H = T + V + 2J - K
            JK = np.einsum("ls,mnls->mn",
                           self.D,
                           2*eri - eri.transpose((0,2,1,3)),
                           optimize=True)

            F_AO = JK + H_core

            F = self.S_inv.T @ F_AO @ self.S_inv            # Orthogonalize fock
            self.MO_energies, self.C_MO = np.linalg.eigh(F) # Diagonalize fock
            C = (self.S_inv @ self.C_MO).real               # Back transform

            # Compute new density
            self.D = np.einsum("mk,nk->mn",C[:,:self.occ],C[:,:self.occ],optimize=True)

            E_elec = np.trace(self.D @ (F_AO + H_core))

            delta_E = abs(E_elec + E_nuc - E_tot)
            E_tot = E_elec + E_nuc # Previous iteration's Etot above

            if (debug):
                print("Iter%3d: %.6f Eh  ΔE=%.6f"%(iteration, E_tot, delta_E))
            if (iteration > 200):
                raise SCFConvergeError("ΔE=%.6f"%delta_E)

        SCF_time = (perf_counter() - start_SCF)/iteration
        print("Average SCF time: %.3fms"%(1000*SCF_time))

        print("\nFinal RHF energy: %.9f"%E_tot)

        # Show that the fock matrix is diagonal in MO basis
        F_MO = np.einsum("mj,ni,mn->ij",C,C,F_AO, optimize=True)
        if (np.abs(F_MO-F_MO.T).sum() > 1e-7):
            print("Fock matrix is not orthogonal in the MO basis!")

        self.C = C
        self.HOMO = C.T[self.occ-1]
        self.LUMO = C.T[self.occ]

        #print(np.array2string(D_0, precision=6, suppress_small=True, max_line_width=200))
        return E_tot, C

    def write_molden(self, filename):
        Z, coords = zip(*self.molecule)
        atom_names = [constants.Z_to_element[a] for a in Z]
        f = open(filename, 'w')

        f.write("[Molden Format]\n[Atoms] (AU)\n")
        for atom, i, Z, coord in zip(atom_names,range(len(Z)), Z, coords):
            f.write(f"{atom} {i} {Z} {coord[0]} {coord[1]} {coord[2]}\n")

        basis = self.basis
        f.write("[GTO]\n")
        shell_label = ['s', 'p', 'd', 'f', 'g']
        for i, atom_name in enumerate(atom_names):
            b_segment = basis[atom_name]
            f.write(f"{i} 0\n")
            for shell, alpha, coeff in b_segment:
                f.write(f"{shell_label[shell]}  {len(coeff.T[0])}  1.00\n")
                for a, c in zip(alpha, coeff.flatten()):
                    f.write(f"  {a}  {c}\n")
            f.write("\n")

        f.write("[MO]\n")
        occupations = 2*np.diag(self.D)
        for e, occ, pii in zip(self.MO_energies, self.C_MO.T, occupations):
            f.write(f" Sym= A\n Ene= {e}\n Spin= Alpha\n Occup= {pii}\n")
            for ao_num, mo_coeff in enumerate(occ):
                f.write(f"   {ao_num+1}  {mo_coeff}\n")

