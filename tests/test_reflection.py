"""Interpolated minimum finding and grid-spacing detection."""
import numpy as np

from reflectometry.reflection import (find_minima_db, global_min_db, minima_from_db,
                                      resolve_spacing)


def test_parabolic_minimum_exact_on_linear_grid():
    f = np.linspace(100e6, 120e6, 201)
    f0, d0, c = 109.3e6 + 1234.5, -7.0, 2e-13
    dB = d0 + c * (f - f0) ** 2
    fm, dm = minima_from_db(f, dB)[0]
    assert abs(fm[0] - f0) < 1e-3          # Hz
    assert abs(dm[0] - d0) < 1e-9          # dB


def test_parabolic_minimum_exact_on_log_grid():
    f = np.geomspace(1e6, 200e6, 4000)
    f0, d0 = 47.3e6, -6.0
    x = np.log10(f)
    dB = d0 + 0.8 * (x - np.log10(f0)) ** 2
    fm, dm = minima_from_db(f, dB, spacing="log")[0]
    assert abs(fm[0] - f0) / f0 < 1e-7
    assert abs(dm[0] - d0) < 1e-9


def test_spacing_autodetect():
    assert resolve_spacing(np.linspace(1e6, 2e8, 100)) == "linear"
    assert resolve_spacing(np.geomspace(1e6, 2e8, 100)) == "log"


def test_spacing_rejects_irregular_grid():
    try:
        resolve_spacing(np.array([1.0, 2.0, 4.0, 5.0]))
    except ValueError:
        return
    raise AssertionError("expected ValueError for an irregular grid")


def test_spacing_explicit_value_is_used():
    assert resolve_spacing(np.array([1.0, 2.0]), spacing="linear") == "linear"


def test_global_min_reports_nan_when_no_dip():
    f = np.linspace(1e6, 2e8, 500)
    Z = np.full((1, f.size), 50.0 + 0j)    # matched everywhere: |Gamma| = 0, no interior minimum
    fm, dm = global_min_db(f, Z, 50.0)[0]
    assert np.isnan(fm) and np.isnan(dm)


def test_find_minima_db_returns_one_entry_per_row():
    f = np.linspace(1e6, 2e8, 500)
    Z = np.vstack([np.full(f.size, 50.0 + 0j), np.full(f.size, 75.0 + 0j)])
    out = find_minima_db(f, Z, 50.0)
    assert set(out.keys()) == {0, 1}
