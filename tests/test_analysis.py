"""Cable identity, impedance shapes, sweeps and minimum pairing."""
import numpy as np

from reflectometry import (BridgeParams, CableParams, compare_minima, input_impedance,
                           resonance_sweep)
from reflectometry.cable import cable_model


def test_zero_length_cable_is_identity():
    abcd, _, _ = cable_model(np.array([1e6, 1e8, 1e9]), CableParams(length=0.0))
    assert np.allclose(abcd[0, 0], 1) and np.allclose(abcd[1, 1], 1)
    assert np.allclose(abcd[0, 1], 0) and np.allclose(abcd[1, 0], 0)


def test_input_impedance_shapes():
    f = np.linspace(1e6, 1e8, 50)
    b = BridgeParams()
    c = CableParams(length=1.0)
    assert input_impedance(f, [75.0, 1000.0], c, b).shape == (2, 50)
    assert input_impedance(f, 75.0, c, b).shape == (50,)
    assert input_impedance(1e7, [75.0, 1000.0], c, b).shape == (2,)
    assert np.shape(input_impedance(1e7, 75.0, c, b)) == ()


def test_resonance_sweep_has_one_entry_per_resistance():
    f = np.geomspace(1e6, 3e8, 3000)
    res = resonance_sweep(f, [75.0, 1000.0, 4000.0], CableParams(length=1.0), BridgeParams())
    assert len(res.resonance_frequencies) == 3
    assert len(res.gamma_at_resonance_db) == 3
    assert np.allclose(res.resistances, [75.0, 1000.0, 4000.0])


def test_compare_minima_pairs_nearest_within_tolerance():
    meas = np.array([50e6, 100e6, 150e6])
    model = np.array([51e6, 120e6])
    c = compare_minima(meas, model, tolerance_hz=5e6)
    assert len(c["pairs"]) == 1
    assert np.isclose(c["pairs"][0][0], 50e6)
    assert np.isclose(c["pairs"][0][1], 51e6)
    assert np.allclose(c["unmatched_measured"], [100e6, 150e6])
    assert np.allclose(c["unmatched_model"], [120e6])
