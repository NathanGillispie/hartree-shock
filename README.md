# ⚠️🤯 ⚡⚡⚡Hartree-Shock ⚡⚡⚡🤯⚠️

Simple Hartree-Fock program to compute my molecular orbitals for me. Final project for programming in chem class.

The `hs` dir contains my final project. Test it by running `pytest` the root dir. It should be used as a module.

**Dependencies**:
 - numpy
 - matplotlib
 - alive-progress
 - pytest
 - gbasis (uses libcint)

[Gbasis](https://github.com/theochem/gbasis) is a super helpful project that computes the integrals using `libcint`. You may need to compile it before ruinning. See the [installation instructions](https://gbasis.qcdevs.org/installation.html) for more info.

## Capabilities

 - Restricted/Unrestricted Hartree-Fock calculations
 - MO ISOSURFACES via plotly (you're welcome)
 - Molden file output (all MOs)
 - Grid file output (one MO)
 - libcint ints OR python ints via gbasis
 - DIIS for faster convergence
 - Cool SCF progress bar, wow!
 - ETA for the SCF procedure 😎

## Options

I use argparse for the options. `--mol` option takes the name of a molecule in the `tests/molecules` dir. Very nice for fast development.

Save and load options for the pickle. Sometimes you just want to replot the MOs and not run the SCF procedure again. Thank me later.

## Why the name?

I thought you'd never ask. The quantum chemistry space is filled with comical names for quantum chemistry software. Check out these classics:

 1. `QChem` from "quantum chemistry"... destroys search engine results when I don't feel like typing "quantum"
 2. `BrianQC` from "Brian quantum chemistry" ...
 3. `Psi4` from ψ. That's like naming CAS software "X"
 4. `Cfour` from "Coupled Cluster techniques for Computational Chemistry"
    Yeah I don't know why it's on this list I just think that's a funny name 💥
 5. `BigDFT` BIG DFT doesn't want you to know these three things!

After acknowledging the absurdity of it all, I present "Hartree-Shock", stylized as above.
