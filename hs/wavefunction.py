
from alive_progress import alive_bar
from alive_progress.animations.bars import bar_factory
from time import perf_counter
import numpy as np
from math import log10
from alive_progress import alive_bar

import ints

class wavefunction:
    """Base class"""

    def __init__(self, molecule, basis, charge=0, multiplicity=1, debug=False, use_libcint=True, integrals={}):
        self.molecule = molecule
        self.basis = basis
        self.charge = charge
        self.mult = multiplicity
        self.e_conv = 1e-6
        self.delta_E = 1
        self.debug = debug

        if integrals=={}:
            mints = ints.integrals(molecule, basis, use_libcint)

            self.nbf = mints.nbf()

            with alive_bar(bar=None, title="Computing Integrals", monitor=False,
                           elapsed=True, stats=False, spinner='arrows_out') as bar:
                E_nuc = ints.nuclear_repulsion(molecule)
                eri = mints.electron_repulsion()
                kinetic = mints.kinetic_energy()
                nuc_attr = mints.nuclear_attraction()
                ovlp = mints.overlap()

            self.ints = {"overlap": ovlp,
                         "kinetic": kinetic,
                         "nuclear": nuc_attr,
                         "eri"    : eri,
                         "E_nuc"  : E_nuc,
                         "nbf"    : self.nbf}
        else:
            self.ints = integrals
            ovlp = self.ints["overlap"]
            self.nbf = self.ints["nbf"]


        # Orthogonalization matrix
        Lambda_S, L_S = np.linalg.eigh(ovlp)
        # \mathbf{S}^{-1/2} aka AO2MO
        self.S_inv = L_S @ np.diag(1/np.sqrt(Lambda_S)) @ np.transpose(L_S)

    def converged(self, delta_E):
        """Is delta_E lower than convergence?"""
        return abs(delta_E) < self.e_conv


class SCF_progress():
    def __init__(self, wfn):
        """
        SCF progress bar. Initializes with wavefunction object.
        Make sure that the initial energy has already been set.
        """
        self.end = -log10(abs(wfn.e_conv))
        self.t_since_SCF = perf_counter()
        self.debug = wfn.debug

    def shocking_bar(self):
        return alive_bar(manual=True, bar=bar_factory(chars='⚡'), length=40)

    def _norm(self, E):
        lE = -log10(E)
        return lE/self.end

    def _clamp01(cls, a):
        if (a < 0):
            return 0
        elif (a > 1):
            return 1
        else:
            return a

    def iter(cls, delta_E):
        return cls._clamp01(cls._norm(abs(delta_E)))

    def run(self, delta_E, E_tot, iteration, bar):
        bar(self.iter(delta_E))
        bar.title=" SCF:%3d"%iteration
        if (self.debug):
            now = perf_counter()
            dt = now - self.t_since_SCF
            self.t_since_SCF = now
            print("Iter%3d: %.6f Eh  ΔE=%.6f  %.2fms"%(iteration, E_tot, abs(self.delta_E), dt*1000. ))


