# bohmian
Seam Geometry of Bohmian Mechanics: Geodesics, Foliations, and Nodal Singularities, a scientific paper.

## Figures

This repo intentionally does not assume a local LaTeX toolchain.

- Create a Python venv and install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Generate the current placeholder foliation figure: `python scripts/harmonic_qd_foliation.py`
- Generate an experimental QD-vs-current comparison (vortex test): `python scripts/qd_vs_current_vortex.py`
- Generate a symmetry-broken vortex sanity check (shifted vortex): `python scripts/qd_vs_current_shifted_vortex.py`
- Generate a limitation/non-example test (two nodes): `python scripts/qd_vs_current_two_nodes.py`

The script writes `figures/bohm_foliation_harmonic.pdf`, which is included by the LaTeX source if present.
