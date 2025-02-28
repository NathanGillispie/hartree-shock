
from time import perf_counter
import numpy as np
from math import log10
# Alternate eigenvalue computer
# from scipy.linalg import eigh

import wavefunction as wfn
from utils import SCFConvergeError
import constants

from diis import DIIS

import matplotlib as mpl

def heatmap(graph, D, labels):
    """Assuming:
    ```py
    import matplotlib.pyplot as plt
    plt.ion()
    fig, graph = plt.subplots()
    fig.tight_layout()
    fig.set_size_inches(8,8)
    ```
    """
    graph.matshow(D, cmap=mpl.colormaps['seismic'], vmin=-3.2, vmax=3.2)

    graph.set_xticks(range(D.shape[0]), labels=labels,
                     ha="right", rotation=-30, rotation_mode="anchor")
    graph.set_yticks(range(D.shape[0]), labels=labels)
    graph.grid(which="minor", color="w", linestyle='-', linewidth=3)

def make_labels(wfn):
    from gbasis.parsers import make_contractions
    basis = wfn.basis
    Z, coords = zip(*wfn.molecule)
    atoms = [constants.Z_to_element[a] for a in Z]
    shells = make_contractions(basis, atoms, np.asarray(coords), coord_types="cartesian")

    labels = []
    L_to_subshell = ["s", "p", "d", "f", "g"]

    for s in shells:
        atom = atoms[s.icenter]
        spdf = L_to_subshell[s.angmom]
        for _ in range(s.num_seg_cont):
            if s.num_cart == 1:
                labels.append(f"{atom}:{spdf}")
            else:
                for i in range(s.num_cart):
                    labels.append(f"{atom}:{spdf}{i}")

    return labels

class UHF(wfn.wavefunction):
    """UHF class, instantiated with molecule, basis and optionally\
    charge and multiplicity."""

    def __init__(self, molecule, basis, plot=False, **kwargs):
        super().__init__(molecule, basis, **kwargs)
        self.plot = plot

        self.occ_a, self.occ_b = self.occupied_orbitals()
        print(f"(occ/nbf) {self.occ_a}/{self.nbf}")

        S_inv = self.S_inv   # Orthogonalization matrix
        H_core = self.H_core = self.ints["kinetic"] + self.ints["nuclear"]
        self.E_nuc = self.ints["E_nuc"]   # Nuclear repulsion

        # Initial fock matrix and MO coefficients using core guess
        # Core guess means F_0' = H_core
        F_0 = S_inv.T @ H_core @ S_inv
        self.MO_energies, self.C_MO = np.linalg.eigh(F_0)
        C_0 = S_inv @ self.C_MO

        # Initial density
        self.D_aa = C_0[:,:self.occ_a] @ C_0[:,:self.occ_a].T
        self.D_bb = C_0[:,:self.occ_b] @ C_0[:,:self.occ_b].T

        # Initial energy (using the core guess)
        E_elec = np.trace(self.D_aa @ (H_core + H_core))
        self.E_tot = E_elec + self.E_nuc

        # Set initial C
        self.C_a = C_0
        self.C_b = C_0

    def compute_E(self, use_diis=True):
        """Computes the RHF energy. Returns (energy, MOs)."""
        start = perf_counter()

        eri = self.ints["eri"]
        H_core = self.H_core
        S_inv = self.S_inv
        D_aa = self.D_aa
        D_bb = self.D_bb
        E_tot = self.E_tot

        if (self.debug):
            print("Initial orbital energies: %sEh"%np.array2string(self.MO_energies,precision=3,max_line_width=150,suppress_small=True,separator=''))
            print("\nIter  0: %.6f Eh"%self.E_tot)

        delta_E = 1
        iteration = 0
        progress = wfn.SCF_progress(self)

        if (use_diis):
            diis_a = DIIS(self, keep=8)
            diis_b = DIIS(self, keep=8)

        if self.plot:
            labels = make_labels(self)
            import matplotlib.pyplot as plt
            plt.ion()
            fig, graph = plt.subplots()
            fig.set_size_inches(8,8)
            heatmap(graph, H_core, labels)
            fig.tight_layout()

        with progress.shocking_bar() as bar:
            while not self.converged(delta_E):
                iteration += 1
                if iteration >= 200:
                    raise SCFConvergeError("ΔE=%.6f"%delta_E)

                J = np.einsum("ls,mnls->mn",(D_aa+D_bb), eri, optimize=True)

                K_a  = np.einsum("ls,mlns->mn",D_aa, eri, optimize=True)
                K_b  = np.einsum("ls,mlns->mn",D_bb, eri, optimize=True)

                self.F_AO_a = H_core + J - K_a
                self.F_AO_b = H_core + J - K_b

                if (use_diis):
                    self.F_AO_a = diis_a.compute_F(self.F_AO_a, D_aa)
                    self.F_AO_b = diis_b.compute_F(self.F_AO_b, D_bb)

                F_a = S_inv.T @ self.F_AO_a @ S_inv              # Orthogonalize fock
                F_b = S_inv.T @ self.F_AO_b @ S_inv              # Orthogonalize fock
                self.MO_energies_a, C_MO_a = np.linalg.eigh(F_a) # Diagonalize fock
                self.MO_energies_b, C_MO_b = np.linalg.eigh(F_b) # Diagonalize fock
                self.C_a = (S_inv @ C_MO_a).real                 # Back transform
                self.C_b = (S_inv @ C_MO_b).real                 # Back transform

                # Compute new density
                D_aa = self.C_a[:,:self.occ_a] @ self.C_a[:,:self.occ_a].T
                D_bb = self.C_b[:,:self.occ_b] @ self.C_b[:,:self.occ_b].T


                E_elec  = np.trace(D_aa @ (self.F_AO_a + H_core))*.5
                E_elec += np.trace(D_bb @ (self.F_AO_b + H_core))*.5

                delta_E = E_elec + self.E_nuc - E_tot
                E_tot = E_elec + self.E_nuc # Previous iteration's Etot above

                progress.run(delta_E, E_tot, iteration, bar)

                if self.plot:
                    heatmap(graph, K, labels)
                    fig.colorbar
                    plt.pause(1)

        E_HOMO = constants.au2ev * self.MO_energies_a[self.occ_a-1]
        E_LUMO = constants.au2ev * self.MO_energies_a[self.occ_a]
        E_GAP  = abs(E_HOMO-E_LUMO)
        print("HOMO/LUMO: Δ(%.4f, %.4f) = %.4feV"%(E_HOMO, E_LUMO, E_GAP), end='')
        if (E_GAP > 1e-20):
            print("   %.2fnm"%(1239.8/E_GAP))
        
        total_spin = np.sum(np.diag(D_aa) - np.diag(D_bb))
        spin_exp = (self.mult-1)*(self.mult+1)/4
        print("⟨S²⟩ = %.6f    Expecting %.6f"%(total_spin, spin_exp))

        print("\nFinal UHF energy (Eh): %.9f"%E_tot)

        self.HOMO = C_MO_a.T[self.occ_a-1]

        self.D_aa = D_aa
        self.D_bb = D_bb
        self.E_tot = E_tot
        self.C_MO_a = C_MO_a
        self.C_MO_b = C_MO_b

        if self.plot:
            input("Press enter to continue...")

        return E_tot, self.C_a, self.C_b

    def occupied_orbitals(self):
        """Returns number of occupied oribitals"""
        Z_total = np.asarray([a[0] for a in self.molecule], int).sum()
        num_elec = Z_total - self.charge
        unpaired_elec = self.mult - 1
        if (num_elec - unpaired_elec)%2 != 0:
            exit("Invalid charge and multiplicity")
        occ_b = (num_elec - unpaired_elec)//2

        # unpaired e always go in a.
        # a is larger
        occ_a = occ_b + unpaired_elec

        if (occ_a > self.nbf):
            exit("More occupied MOs than basis functions. This should never happen. Good luck.")
        if (occ_a== self.nbf):
            exit("No LUMO orbital, choose a large basis set")
        return occ_a, occ_b

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
        occupations = np.diag(self.D_aa)
        for e, occ, pii in zip(self.MO_energies_a, self.C_MO_a.T, occupations):
            f.write(f" Sym= A\n Ene= {e:.6f}\n Spin=Alpha\n Occup= {pii:.6f}\n")
            for ao_num, mo_coeff in enumerate(occ):
                f.write(f"   {ao_num+1}  {mo_coeff}\n")
        occupations = np.diag(self.D_bb)
        for e, occ, pii in zip(self.MO_energies_b, self.C_MO_b.T, occupations):
            f.write(f" Sym= B\n Ene= {e:.6f}\n Spin=Beta\n Occup={pii:.6f}\n")
            for ao_num, mo_coeff in enumerate(occ):
                f.write(f"   {ao_num+1}  {mo_coeff}\n")


