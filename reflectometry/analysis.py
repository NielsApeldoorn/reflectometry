"""
Resonance analysis of the cable-plus-bridge system, including parametric sweeps.

The main entry points are:

* ``resonance_sweep``: resonance frequency and depth versus load resistance,
  for one set of cable and bridge parameters.
* ``parametric_resonance_sweep``: the same, repeated over a grid of any cable or bridge
  parameters (cartesian product of the values given).
"""
import sys
import time
import itertools
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, List, Sequence

import numpy as np

from .cable import CableParams, cable_model
from .circuit import BridgeParams, bridge_matrix, cascade
from .reflection import (find_minima_db, global_min_db, reflection_db,
                         resolve_spacing, to_axis, from_axis)


def default_frequencies(log10_start=6.0, log10_stop=8.6, n=10000):
    """Log-spaced frequency grid in Hz (default: 10 MHz to ~400 MHz)."""
    return np.logspace(log10_start, log10_stop, n)


@dataclass
class ResonanceSweep:
    """Resonance versus load resistance."""

    resistances: np.ndarray            # ohm
    resonance_frequencies: np.ndarray  # Hz; NaN where no interior minimum exists
    gamma_at_resonance_db: np.ndarray  # dB at the resonance minimum
    matching_resistances: np.ndarray = None  # ohm; optional, where Z_in = Z0
    matching_frequencies: np.ndarray = None  # Hz; optional, where Z_in = Z0


def input_impedance(f, R, cable=CableParams(), bridge=BridgeParams()):
    """
    Impedance seen at the VNA for one or more load resistances ``R`` (ohm).

    The output shape is np.shape(R) + np.shape(f), so a scalar input adds no axis:
      - array R, array f  -> (nR, N)
      - scalar R, array f -> (N,)
      - array R, scalar f -> (nR,)
      - scalar R, scalar f -> 0-d array
    """
    f_arr = np.asarray(f, dtype=float)
    R_arr = np.asarray(R, dtype=float)

    f1 = np.atleast_1d(f_arr)                  # the internal routines need an array
    abcd, _, _ = cable_model(f1, cable)
    T = cascade(abcd, bridge_matrix(f1, bridge))
    A, B = T[0, 0], T[0, 1]
    C, D = T[1, 0], T[1, 1]

    Rc = R_arr.reshape(-1, 1)                  # (nR, 1)
    Z = (A * Rc + B) / (C * Rc + D)            # (nR, N)
    return Z.reshape(R_arr.shape + f_arr.shape)


def reflection_spectrum_db(f, resistance, cable=CableParams(), bridge=BridgeParams()):
    """|Gamma| in dB versus frequency for one load resistance."""
    Z = input_impedance(f, [resistance], cable, bridge)[0]
    return reflection_db(Z, cable.Z0c)


@dataclass
class ParametricPoint:
    params: Dict[str, Any]
    result: ResonanceSweep
    cable: CableParams = field(default_factory=CableParams)
    bridge: BridgeParams = field(default_factory=BridgeParams)

def flatten_minima(sweep):
    """
    All minima as flat arrays, one entry per (resistance, minimum) pair.
    Returns (resistances_ohm, frequencies_Hz, gamma_db).
    """
    R, F, G = [], [], []
    for r, fk, gk in zip(sweep.resistances,
                         sweep.resonance_frequencies,
                         sweep.gamma_at_resonance_db):
        fk = np.atleast_1d(fk)
        gk = np.atleast_1d(gk)
        R.append(np.full(len(fk), r))
        F.append(fk)
        G.append(gk)
    if not R:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(R), np.concatenate(F), np.concatenate(G)

def resonance_sweep(f, resistances, cable=CableParams(), bridge=BridgeParams(), spacing="auto"):
    """
    Find the reflection minimum seen by the VNA for each load resistance.

    Parameters
    ----------
    f : (N,) array
        Frequency grid in Hz, uniform in log10(f) or in f.
    resistances : array_like
        Load resistances in ohm.
    cable : CableParams
    bridge : BridgeParams
    spacing : {"auto", "log", "linear"}
        Spacing of ``f``. "auto" detects it from the grid.

    Returns
    -------
    ResonanceSweep
    """
    R = np.atleast_1d(np.asarray(resistances, dtype=float))
    Z = input_impedance(f, R, cable, bridge)                    # (nR, N) impedance at VNA

    minima = find_minima_db(f, Z, cable.Z0c, spacing=spacing)
    freqs = [minima[k][0] for k in range(len(R))]
    gammas = [minima[k][1] for k in range(len(R))]

    return ResonanceSweep(
        resistances=R,
        resonance_frequencies=freqs,
        gamma_at_resonance_db=gammas,
    )

def matching_resistances(f=None, Z0=50.0, cable=CableParams(), bridge=BridgeParams(),
                         spacing="auto"):
    """
    Frequencies at which a real, positive load resistance gives Z_in = Z0
    (no reflection), and the resistance at each.

    Returns (frequencies_Hz, resistances_ohm), interpolated at sign changes
    of Im(R(f)), linearly in the grid's coordinate (log10 f or f).
    """
    if f is None:
        f = default_frequencies()
    f = np.asarray(f, dtype=float)
    spacing = resolve_spacing(f, spacing)
    abcd, _, _ = cable_model(f, cable)
    T = cascade(abcd, bridge_matrix(f, bridge))
    A, B = T[0, 0], T[0, 1]
    C, D = T[1, 0], T[1, 1]
    R = (Z0 * D - B) / (A - Z0 * C)

    im = R.imag
    idx = np.nonzero(np.sign(im[:-1]) * np.sign(im[1:]) < 0)[0]
    x = to_axis(f, spacing)
    fs, Rs = [], []
    for k in idx:
        t = im[k] / (im[k] - im[k + 1])
        fm = from_axis(x[k] + t * (x[k + 1] - x[k]), spacing)
        Rm = R[k].real + t * (R[k + 1].real - R[k].real)
        if Rm > 0:
            fs.append(fm); Rs.append(Rm)
    return np.array(fs), np.array(Rs)

def _im_z_exact(x, R, cable, bridge, spacing):
    """Im(Z_in) at grid coordinate x (log10 f or f, per spacing) for a single load resistance R."""
    return float(np.imag(input_impedance(from_axis(x, spacing), R, cable, bridge)))

def zero_reactance_frequencies(R_eq, f=None, cable=CableParams(), bridge=BridgeParams(),
                               refine=True, n_bisect=60, plot=False, spacing="auto"):
    """
    Frequencies (Hz) where Im(Z_in) = 0 for one or more load resistances.

    R_eq : float or array_like
        A scalar returns one array of roots. An array returns a list with one
        array of roots per resistance. The counts can differ, so the arrays
        can't be stacked into a single 2-D array.
    spacing : {"auto", "log", "linear"}
        Spacing of ``f``. Roots are located and interpolated in that coordinate.
    """
    scalar = np.ndim(R_eq) == 0
    R_arr = np.atleast_1d(np.asarray(R_eq, dtype=float))
    if f is None:
        f = default_frequencies()
    f = np.asarray(f, dtype=float)
    spacing = resolve_spacing(f, spacing)
    x = to_axis(f, spacing)

    im_all = input_impedance(f, R_arr, cable, bridge).imag   # shape (nR, N)

    results = []
    for i, R in enumerate(R_arr):
        im = im_all[i]
        idx = np.nonzero(np.sign(im[:-1]) * np.sign(im[1:]) < 0)[0]
        roots = []
        for k in idx:
            a, b = x[k], x[k + 1]
            if refine:
                ga = _im_z_exact(a, R, cable, bridge, spacing)
                for _ in range(n_bisect):
                    m = 0.5 * (a + b)
                    gm = _im_z_exact(m, R, cable, bridge, spacing)
                    if np.sign(gm) == np.sign(ga):
                        a, ga = m, gm
                    else:
                        b = m
                roots.append(from_axis(0.5 * (a + b), spacing))
            else:
                t = im[k] / (im[k] - im[k + 1])
                roots.append(from_axis(a + t * (b - a), spacing))
        results.append(np.array(roots))

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, R in enumerate(R_arr):
            line, = ax.semilogx(f, im_all[i], label=f"R_eq={R:g} ohm")
            ax.scatter(results[i], np.zeros(len(results[i])),
                       color=line.get_color(), marker="x", s=30, zorder=3)
        ax.axhline(0, color="k", linestyle="--")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Im(Z) (ohm)")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        fig.tight_layout()

    return results[0] if scalar else results

def parametric_resonance_sweep(resistances, grid, f=None,
                     cable=CableParams(), bridge=BridgeParams(), spacing="auto"):
    """
    Run ``resonance_sweep`` over every combination of the parameter values in ``grid``.

    Parameters
    ----------
    resistances : array_like
        Load resistances in ohm, shared by all runs.
    grid : dict
        Maps parameter names to iterables of values. Names must be fields of
        ``CableParams`` or ``BridgeParams``, for example::

            {"length": [1.0, 2.0, 3.0], "C_T": [40e-12, 56e-12]}

    f : (N,) array, optional
        Frequency grid. Defaults to ``default_frequencies()``.
    cable, bridge : optional
        Base parameters. Values in ``grid`` override these.

    Returns
    -------
    list of ParametricPoint
        One entry per combination, in the order produced by
        ``itertools.product`` over the grid's keys.
    """
    if f is None:
        f = default_frequencies()

    cable_names = {fld.name for fld in fields(CableParams)}
    bridge_names = {fld.name for fld in fields(BridgeParams)}
    unknown = set(grid) - cable_names - bridge_names
    if unknown:
        raise KeyError(
            f"Unknown parameter(s) {sorted(unknown)}. Valid names are "
            f"{sorted(cable_names | bridge_names)}."
        )

    names: List[str] = list(grid)
    value_lists: Sequence[list] = [list(grid[name]) for name in names]

    points = []
    for combo in itertools.product(*value_lists):
        assignment = dict(zip(names, combo))
        cable_over = {k: v for k, v in assignment.items() if k in cable_names}
        bridge_over = {k: v for k, v in assignment.items() if k in bridge_names}

        result = resonance_sweep(
            f, resistances,
            cable=replace(cable, **cable_over),
            bridge=replace(bridge, **bridge_over),
            spacing=spacing,
        )
        c = replace(cable, **cable_over)
        result.matching_frequencies, result.matching_resistances = matching_resistances(
            f, Z0=c.Z0c, cable=c, bridge=replace(bridge, **bridge_over), spacing=spacing
        )

        points.append(ParametricPoint(
            params=assignment,
            result=result,
            cable=c,
            bridge=replace(bridge, **bridge_over),
        ))
    return points

def _print_progress(done, total, start, width=30):
    """Overwrite one console line with a progress bar, count, elapsed time and ETA."""
    frac = done / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.perf_counter() - start
    eta = elapsed / done * (total - done) if done else 0.0
    sys.stdout.write(
        f"\r[{bar}] {done}/{total} ({100 * frac:5.1f}%)  "
        f"elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s"
    )
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def parametric_peak_sweep(grid, f=None,
                          cable=CableParams(), bridge=BridgeParams(),
                          progress=True, spacing="auto"):
    """
    Run ``zero_reactance_frequencies`` over every combination of the parameter values in ``grid``.

    The load resistance for each point is that point's ``R_eq``, so ``R_eq``
    can be included in ``grid`` like any other bridge parameter.
    ...
    """
    if f is None:
        f = default_frequencies()

    cable_names = {fld.name for fld in fields(CableParams)}
    bridge_names = {fld.name for fld in fields(BridgeParams)}
    unknown = set(grid) - cable_names - bridge_names
    if unknown:
        raise KeyError(
            f"Unknown parameter(s) {sorted(unknown)}. Valid names are "
            f"{sorted(cable_names | bridge_names)}."
        )

    names: List[str] = list(grid)
    value_lists: Sequence[list] = [list(grid[name]) for name in names]
    combos = list(itertools.product(*value_lists))
    total = len(combos)

    points = []
    start = time.perf_counter()
    if progress:
        _print_progress(0, total, start)

    for done, combo in enumerate(combos, start=1):
        assignment = dict(zip(names, combo))
        cable_over = {k: v for k, v in assignment.items() if k in cable_names}
        bridge_over = {k: v for k, v in assignment.items() if k in bridge_names}
        c = replace(cable, **cable_over)
        b = replace(bridge, **bridge_over)

        load = [b.R_eq]                       # this point's load resistance
        result = zero_reactance_frequencies(load, f, cable=c, bridge=b, spacing=spacing)
        result_dict = {r: f_r for r, f_r in zip(load, result)}

        points.append(ParametricPoint(params=assignment, result=result_dict,
                                      cable=c, bridge=b))

        if progress:
            _print_progress(done, total, start)

    return points


def compare_minima(measured_f, model_f, tolerance_hz=5e6):
    """
    Pair measured dips with model dips by nearest frequency.

    Pairing is greedy: each measured dip, in frequency order, takes the nearest
    unused model dip within ``tolerance_hz``. Returns a dict with:

    * ``pairs``: list of ``(f_measured, f_model, f_model - f_measured)`` in Hz
    * ``unmatched_measured``: measured dips with no model partner
    * ``unmatched_model``: model dips with no measured partner
    """
    measured = np.sort(np.asarray(measured_f, dtype=float))
    model = np.sort(np.asarray(model_f, dtype=float))
    used = np.zeros(len(model), dtype=bool)
    pairs, unmatched_meas = [], []

    for fm in measured:
        cand = np.where(~used)[0]
        if len(cand) == 0:
            unmatched_meas.append(fm)
            continue
        j = cand[np.argmin(np.abs(model[cand] - fm))]
        if abs(model[j] - fm) <= tolerance_hz:
            used[j] = True
            pairs.append((fm, model[j], model[j] - fm))
        else:
            unmatched_meas.append(fm)

    return {
        "pairs": pairs,
        "unmatched_measured": np.array(unmatched_meas),
        "unmatched_model": model[~used],
    }
