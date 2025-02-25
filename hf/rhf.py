
from time import perf_counter
import numpy as np
# Alternate eigenvalue computer
# from scipy.linalg import eigh

import ints
import wavefunction as wfn
from utils import SCFConvergeError
import constants
from math import log10

from diis import DIIS

class RHF(wfn.wavefunction):
    """RHF class, instantiated with molecule, basis and optionally\
    charge and multiplicity."""

    def __init__(self, molecule, basis):
        super().__init__(molecule, basis)
        occ = self.occupied_orbitals()
        print(f"(occ/nbf) {occ}/{self.nbf}")

        S_inv = self.S_inv   # Orthogonalization matrix
        H_core = self.H_core # Kinetic + nuc_attr
        E_nuc = self.E_nuc   # Nuclear repulsion

        # Initial fock matrix and MO coefficients using core guess
        # Core guess means F_0' = H_core
        F_0 = S_inv.T @ H_core @ S_inv
        self.MO_energies, self.C_MO = np.linalg.eigh(F_0)
        C_0 = S_inv @ self.C_MO

        # Initial density
        self.D = np.einsum("mk,nk->mn",C_0[:,:occ],C_0[:,:occ],optimize=True)

        # Initial energy (using the core guess)
        E_elec = np.trace(self.D @ (H_core + H_core))
        self.E_tot = E_elec + E_nuc

        # Set initial C
        self.C = C_0
        self.HOMO = self.C.T[occ-1]
        self.LUMO = self.C.T[occ]
        self.occ = occ

    def compute_E(self, use_diis=True):
        """Computes the RHF energy. Returns (energy, MOs)."""
        start = perf_counter()

        eri = self.ints["eri"]
        H_core = self.H_core
        S_inv = self.S_inv
        E_nuc = self.E_nuc
        occ = self.occ
        D = self.D
        E_tot = self.E_tot
        C_MO = self.C_MO

        if (self.debug):
            print("Initial orbital energies: %sEh"%np.array2string(self.MO_energies,precision=3,max_line_width=150,suppress_small=True,separator=''))
            print("\nIter  0: %.6f Eh"%self.E_tot)

        delta_E = 1
        iteration = 0
        progress = wfn.SCF_progress(self)

        if (use_diis):
            diis = DIIS(self, keep=8)

        with progress.shocking_bar() as bar:
            while not self.converged(delta_E):
                iteration += 1
                if iteration >= 200:
                    raise SCFConvergeError("ΔE=%.6f"%delta_E)

                JK = np.einsum("ls,mnls->mn", D,
                               2*eri - eri.transpose((0,2,1,3)),
                               optimize=True)

                self.F_AO = JK + H_core

                # Warning: using previous Density for diis
                if (use_diis):
                    self.F_AO = diis.compute_F(self.F_AO, D)

                F = S_inv.T @ self.F_AO @ S_inv            # Orthogonalize fock
                self.MO_energies, C_MO = np.linalg.eigh(F) # Diagonalize fock
                self.C = (S_inv @ C_MO).real               # Back transform

                # Compute new density
                D = np.einsum("mk,nk->mn",self.C[:,:occ],self.C[:,:occ],optimize=True)

                E_elec = np.trace(D @ (self.F_AO + H_core))

                delta_E = E_elec + E_nuc - E_tot
                E_tot = E_elec + E_nuc # Previous iteration's Etot above

                progress.run(delta_E, E_tot, iteration, bar)

        E_HOMO = constants.au2ev * self.MO_energies[occ-1]
        E_LUMO = constants.au2ev * self.MO_energies[occ]
        E_GAP  = abs(E_HOMO-E_LUMO)
        print("HOMO/LUMO: Δ(%.4f, %.4f) = %.4feV"%(E_HOMO, E_LUMO, E_GAP), end='')
        if (E_GAP > 1e-20):
            print("   %.2fnm"%(1239.8/E_GAP))
        print("\nFinal RHF energy (Eh): %.9f"%E_tot)

        self.HOMO = self.C.T[occ-1]
        self.LUMO = self.C.T[occ]
        self.D = D
        self.E_tot = E_tot
        self.C_MO = C_MO

        #print(np.array2string(D_0, precision=6, suppress_small=True, max_line_width=200))
        return E_tot, self.C

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

    def write_molden(self, file):
        """Necessary to view MOs"""
        Z, coords = zip(*self.molecule)
        atom_names = [constants.Z_to_element[a] for a in Z]

        f = file
        if type(file) == str:
            f = open(f, 'w')

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

