"""Impedance, reflection coefficient, and resonance (minimum) detection."""

import numpy as np


def resolve_spacing(f, spacing="auto"):
    """
    Return ``"log"`` or ``"linear"``: how the frequency grid ``f`` is spaced.

    With ``"auto"``, the grid is classified as uniform in log10(f) or uniform in f.
    A grid that is neither, or that has fewer than three points, needs ``spacing``
    given explicitly.
    """
    if spacing in ("log", "linear"):
        return spacing
    if spacing != "auto":
        raise ValueError(f"spacing must be 'auto', 'log' or 'linear', not {spacing!r}")

    f = np.asarray(f, dtype=float)
    if f.ndim != 1 or len(f) < 3:
        raise ValueError("Cannot infer grid spacing from fewer than 3 points; "
                         "pass spacing='log' or spacing='linear'.")

    def uniform(x):
        d = np.diff(x)
        return np.allclose(d, d.mean(), rtol=1e-6, atol=0.0)

    if uniform(np.log10(f)):
        return "log"
    if uniform(f):
        return "linear"
    raise ValueError("Frequency grid is neither uniform in log10(f) nor in f; "
                     "pass spacing='log' or spacing='linear' explicitly.")


def to_axis(f, spacing):
    """Coordinate in which the grid is uniform: log10(f) for a log grid, f for a linear one."""
    f = np.asarray(f, dtype=float)
    return np.log10(f) if spacing == "log" else f


def from_axis(x, spacing):
    """Frequency in Hz from a grid coordinate. Inverse of ``to_axis``."""
    return 10.0 ** x if spacing == "log" else x


def load_impedance(matrix, load):
    """
    Impedance seen at the input of a two-port terminated by ``load``.

    ``matrix`` is a (2, 2, N) stack; ``load`` is a scalar or broadcastable array.
    """
    return ((matrix[0, 0, :] * load + matrix[0, 1, :])
            / (matrix[1, 0, :] * load + matrix[1, 1, :]))


def reflection_coefficient(Z, Z0c):
    """Complex reflection coefficient of impedance ``Z`` against reference ``Z0c``."""
    return (Z - Z0c) / (Z + Z0c)


def reflection_db(Z, Z0c):
    """Reflection magnitude in dB, 20*log10(|Gamma|)."""
    return 20 * np.log10(np.abs(reflection_coefficient(Z, Z0c)))


def find_minima_db(f, Z, Z0c, eps=1e-30, spacing="auto"):
    """
    Interpolated local minima of 20*log10(|(Z - Z0c)/(Z + Z0c)|).

    Parameters
    ----------
    f : (N,) array
        Frequency grid in Hz, uniform in log10(f) or in f.
    Z : (nR, N) array
        Impedance seen by the VNA, one row per load resistance.
    Z0c : float
        Reference impedance (ohm).
    eps : float
        Floor applied before taking the log, so exact zeros stay finite.
    spacing : {"auto", "log", "linear"}
        Spacing of ``f``. The parabolic interpolation is done in that coordinate.
        "auto" detects it from the grid.

    Returns
    -------
    dict
        Maps row index to a tuple ``(f_min, dB_min)`` of arrays holding the
        interpolated minimum frequencies and values for that row.
    """
    G = np.abs(reflection_coefficient(Z, Z0c))
    dB = 20 * np.log10(np.maximum(G, eps))
    return minima_from_db(f, dB, spacing=spacing)


def minima_from_db(f, dB, spacing="auto"):
    """
    Interpolated local minima of values already in dB, such as a measured S11 trace.

    Parameters
    ----------
    f : (N,) array
        Frequency grid in Hz, uniform in log10(f) or in f.
    dB : (N,) or (nR, N) array
        Values in dB, one row per trace.
    spacing : {"auto", "log", "linear"}
        Spacing of ``f``. "auto" detects it from the grid.

    Returns
    -------
    dict
        Maps row index to ``(f_min, dB_min)`` arrays of the interpolated minima.
    """
    f = np.asarray(f, dtype=float)
    spacing = resolve_spacing(f, spacing)
    x = to_axis(f, spacing)
    dB = np.atleast_2d(np.asarray(dB, dtype=float))

    # Interior local minima: dB[k] < left neighbour and dB[k] <= right neighbour
    y0, y1, y2 = dB[:, :-2], dB[:, 1:-1], dB[:, 2:]
    is_min = (y1 < y0) & (y1 <= y2)
    ii, jj = np.nonzero(is_min)
    k = jj + 1                                   # index of the centre sample

    # Parabola through (k-1, k, k+1) in the grid coordinate
    a = dB[ii, k - 1]
    b = dB[ii, k]
    c = dB[ii, k + 1]
    den = a - 2 * b + c
    p = np.where(den != 0, 0.5 * (a - c) / np.where(den != 0, den, 1.0), 0.0)
    p = np.clip(p, -0.5, 0.5)                    # vertex must stay in the bracket

    h = 0.5 * (x[k + 1] - x[k - 1])              # local grid step in the coordinate
    f_min = from_axis(x[k] + p * h, spacing)
    dB_min = b - 0.25 * (a - c) * p              # parabola value at the vertex

    # np.nonzero returns entries sorted by row, so each row is a contiguous slice
    out = {}
    bounds = np.searchsorted(ii, np.arange(dB.shape[0] + 1))
    for r in range(dB.shape[0]):
        sl = slice(bounds[r], bounds[r + 1])
        out[r] = (f_min[sl], dB_min[sl])
    return out


def global_min_db(f, Z, Z0c, eps=1e-30, spacing="auto"):
    """
    Deepest interpolated minimum per row.

    Returns a dict mapping row index to ``(f_at_min, dB_at_min)``. Rows with no
    interior minimum map to ``(nan, nan)``.
    """
    minima = find_minima_db(f, Z, Z0c, eps=eps, spacing=spacing)
    out = {}
    for r, (fm, dBm) in minima.items():
        if len(dBm) == 0:
            out[r] = (np.nan, np.nan)
        else:
            j = np.argmin(dBm)
            out[r] = (fm[j], dBm[j])
    return out
