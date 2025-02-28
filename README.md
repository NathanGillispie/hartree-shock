# ⚠️🤯 ⚡⚡⚡Hartree-Shock ⚡⚡⚡🤯⚠️
▞▞▞▞▞▞▞▞▞▞▞▞▞▞

Simple Hartree-Fock program to compute my molecular orbitals for me. Final project for programming in chem class.

The `hs` dir contains my final project. Test it by running `pytest` the root dir.

I'm still kinda caught between using this as a module and as a script with `./hf.py`. Bear with me while I transition to making this just a module.

## Capabilities

 - Restricted/Unrestricted Hartree-Fock calculations
 - MO ISOSURFACES via plotly (you're welcome)
 - Molden file output (all MOs)
 - Grid file output (one MO)
 - libcint ints OR python ints via gbasis
 - DIIS for faster convergence
 - Cool SCF progress bar, wow!
 - The progress bar is actually cool though, like I even put in an ETA for the SCF procedure I mean who does that right isn't that the coolest thing you've ever seen? It also has these bolt emoji too and it's super low latency just trust me please.

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
