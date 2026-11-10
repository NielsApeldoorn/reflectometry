# reflectometry

Tools for modelling a VNA-measured reflection-coefficient bridge: cable and bridge
circuit models, resonance detection, and fitting of bridge and cable parameters to
measured S11 data.

## Layout

- `reflectometry/cable.py`: transmission-line model of the coaxial cable (`CableParams`, `cable_model`)
- `reflectometry/circuit.py`: bridge network (`BridgeParams`, `bridge_matrix`, `cascade`)
- `reflectometry/reflection.py`: reflection coefficient and minimum detection
- `reflectometry/analysis.py`: resonance sweeps, matching resistances, parametric sweeps
- `reflectometry/vna.py`: loading VNA CSV exports (`load_vna_csv`, `VnaTrace`)
- `reflectometry/fitting.py`: fitting bridge and cable parameters to data
- `reflectometry/plotting.py`: plotting helpers (matplotlib)
- `reflectometry/experimental.py`: bench reference values

## Installation

    python -m venv .venv
    source .venv/bin/activate
    pip install numpy scipy matplotlib

Run from the repository root so that `import reflectometry` resolves.

## Data

VNA CSV files are not stored in this repository. Keep them in a `data/` directory
(ignored by git) or point `load_vna_csv` at their location.

## Tests

    pip install pytest
    pytest

Without pytest, run `python tests/run_without_pytest.py` from the repository root.

## Scripts

Reproduce the analysis runs from the command line. Put the VNA exports in `data/`
(not tracked by git).

    python scripts/fit_short_cable.py data/cable.csv --band 0.1 200 --plot cable_fit.png
    python scripts/fit_bridge_spectra.py --data-dir data --fit C_P L_P R_L --fit-cable-length --plot bridge_fit.png
