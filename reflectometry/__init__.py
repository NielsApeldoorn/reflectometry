"""
Reflectometry model of a coaxial cable feeding a resonant bridge load.

Typical use::

    from reflectometry import CableParams, BridgeParams, resonance_sweep
    from reflectometry.analysis import default_frequencies

    f = default_frequencies()
    R = np.linspace(75, 10000, 1000)
    result = resonance_sweep(f, R, CableParams(length=3.0), BridgeParams())

Plotting lives in ``reflectometry.plotting`` and is not imported here, so the
computational part of the package works without matplotlib.
"""

from .cable import CableParams, cable_model
from .circuit import BridgeParams, bridge_matrix, cascade
from .reflection import (
    find_minima_db,
    global_min_db,
    load_impedance,
    reflection_coefficient,
    reflection_db,
)
from .analysis import (
    ParametricPoint,
    ResonanceSweep,
    default_frequencies,
    flatten_minima,
    input_impedance,
    matching_resistances,
    parametric_peak_sweep,
    parametric_resonance_sweep,
    resonance_sweep,
    compare_minima,
    zero_reactance_frequencies,
)
from .quadrature import quadratures, resonant_frequency, series_rlc_impedance
from .vna import VnaTrace, find_trace_minima, load_vna_csv
from .fitting import (FitResult, SpectrumFitResult, ZERO_CABLE, CABLE_FIT_PARAMS, fit_bridge, fit_bridge_multistart,
                      fit_spectra, fit_spectra_multistart, measured_resonances, model_frequencies,
                      spectrum_model_db, CableFitResult, CABLE_SHORT_PARAMS, fit_short_cable,
                      short_cable_model_db)

__all__ = [
    "CableParams",
    "cable_model",
    "BridgeParams",
    "bridge_matrix",
    "cascade",
    "load_impedance",
    "reflection_coefficient",
    "reflection_db",
    "find_minima_db",
    "global_min_db",
    "default_frequencies",
    "ResonanceSweep",
    "ParametricPoint",
    "resonance_sweep",
    "parametric_resonance_sweep",
    "parametric_peak_sweep",
    "input_impedance",
    "matching_resistances",
    "zero_reactance_frequencies",
    "flatten_minima",
    "quadratures",
    "resonant_frequency",
    "series_rlc_impedance",
    "VnaTrace",
    "load_vna_csv",
    "find_trace_minima",
    "compare_minima",
    "FitResult",
    "fit_bridge",
    "fit_bridge_multistart",
    "measured_resonances",
    "model_frequencies",
    "SpectrumFitResult",
    "ZERO_CABLE",
    "CABLE_FIT_PARAMS",
    "fit_spectra",
    "fit_spectra_multistart",
    "spectrum_model_db",
    "CableFitResult",
    "CABLE_SHORT_PARAMS",
    "fit_short_cable",
    "short_cable_model_db",
]
