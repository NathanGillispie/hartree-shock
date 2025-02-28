# Hartree-Shock ⚡

Simple Hartree-Fock program to compute my molecular orbitals for me. Final project for programming in chem class.

The `hs` dir contains my final project. Test it by running `pytest` in that dir.

```
  ▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞
  ▞▞ HARTREE SHOCK ⚡▞▞
  ▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞▞
```

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
