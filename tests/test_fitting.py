"""Synthetic recovery checks for the three fitters.

Each test builds data from known parameters, fits from the nominal start, and
checks that the known values come back. Noise is seeded for repeatability.
"""
from dataclasses import replace

import numpy as np

from reflectometry import (BridgeParams, CableParams, ZERO_CABLE, fit_bridge,
                           fit_short_cable, fit_spectra, model_frequencies,
                           short_cable_model_db)
from reflectometry.analysis import reflection_spectrum_db
from reflectometry.vna import VnaTrace


def test_fit_bridge_recovers_reactive_components():
    R = np.array([1147.0, 2144.0, 2684.0, 3223.0, 3497.0, 3749.0, 3950.0, 4312.0])
    f_meas0 = np.full(R.size, 105e6)
    truth = replace(BridgeParams(), C_T=56e-12 * 1.25, L=470e-9 * 0.8,
                    C_P=6e-12 * 1.3, L_P=2e-6 * 0.9)
    f_syn = model_frequencies(R, f_meas0, cable=ZERO_CABLE, bridge=truth,
                              f=np.geomspace(5e6, 300e6, 4000))
    ok = np.isfinite(f_syn)
    res = fit_bridge(R[ok], f_syn[ok], cable=ZERO_CABLE,
                     fit_params=("C_T", "L", "C_P", "L_P"), log_span=1.0, max_nfev=2000)
    for k in ("C_T", "L", "C_P", "L_P"):
        assert abs(res.params[k] / getattr(truth, k) - 1) < 0.01, k


def test_fit_spectra_recovers_bridge_from_noisy_traces():
    freqs = np.linspace(1e5, 2e8, 1000)
    Rs = [1147.0, 2144.0, 2684.0, 3223.0, 3497.0, 3749.0]
    truth = replace(BridgeParams(), C_T=56e-12 * 1.25, L=470e-9 * 0.8,
                    C_P=6e-12 * 1.3, L_P=2e-6 * 0.9)
    rng = np.random.default_rng(3)
    traces = []
    for R in Rs:
        g = reflection_spectrum_db(freqs, R, ZERO_CABLE, truth) + 0.02 * rng.standard_normal(freqs.size)
        traces.append(VnaTrace(freq_hz=freqs, s11_db=g, name=f"syn{R}", resistance_ohm=float(R)))
    res = fit_spectra(traces, fit_params=("C_T", "L", "C_P", "L_P"), start=BridgeParams(),
                      cable=ZERO_CABLE, max_nfev=2000)
    for k in ("C_T", "L", "C_P", "L_P"):
        assert abs(res.params[k] / getattr(truth, k) - 1) < 0.05, k


def test_fit_short_cable_recovers_length_and_losses():
    f = np.linspace(1e5, 6e9, 1000)
    truth = CableParams(length=1.37, loss_sqrt=2.9, loss_lin=0.0045)
    rng = np.random.default_rng(5)
    g = short_cable_model_db(f, truth) + 0.02 * rng.standard_normal(f.size)
    res = fit_short_cable(f, g, start=CableParams(length=1.0))
    assert abs(res.params["length"] / truth.length - 1) < 0.01
    assert abs(res.params["loss_sqrt"] / truth.loss_sqrt - 1) < 0.05
    assert abs(res.params["loss_lin"] / truth.loss_lin - 1) < 0.05
