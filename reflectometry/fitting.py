"""
Fit bridge parameters to measured resonances. The cable is held fixed.

Each measured point is a load resistance, the frequency of its resonance dip,
and optionally the dip depth. For a trial set of bridge parameters, the model
dip at each resistance is the minimum nearest the measured frequency. The fit
minimises the relative frequency error, plus an optional depth term.

Parameters are fitted in log10 space, so they stay positive and the optimiser
steps by roughly the same relative amount for every component.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from scipy.optimize import differential_evolution

from .analysis import reflection_spectrum_db, resonance_sweep
from .cable import cable_model
from .reflection import reflection_db
from .vna import VnaTrace
from .cable import CableParams
from .circuit import BridgeParams
from .experimental import MEASURED_RESONANCES

# Components varied by default. The resonance frequencies depend on the reactive
# elements; R_T and R_L mostly set the dip depth, so fit them only when depth data
# is supplied (depth_weight > 0). R_eq is the plotting termination, not fitted.
DEFAULT_FIT_PARAMS = ("C_T", "L", "C_P", "L_P")

# Cable used for spectrum fits: zero length, so the model is the bridge alone.
ZERO_CABLE = CableParams(length=0.0)

# Residual used when the model has no dip at a measured resistance at all.
NO_DIP_PENALTY = 1.0


@dataclass
class FitResult:
    """Outcome of ``fit_bridge``."""

    bridge: BridgeParams                  # bridge with the fitted values
    params: Dict[str, float]              # fitted values, SI units
    stderr: Dict[str, float]              # approximate 1-sigma, SI units (nan if undefined)
    resistances: np.ndarray               # ohm, as fitted
    measured_frequencies_hz: np.ndarray   # Hz, as fitted
    model_frequencies_hz: np.ndarray      # Hz, model dip nearest each measurement
    residuals: np.ndarray
    rms_relative_error: float             # RMS of (f_model - f_meas) / f_meas
    cost: float                           # 0.5 * sum(residuals**2)
    success: bool
    message: str
    nfev: int = 0
    fit_params: Tuple[str, ...] = field(default=DEFAULT_FIT_PARAMS)


def measured_resonances(table=MEASURED_RESONANCES):
    """
    The bench table as arrays: ``(resistances_ohm, frequencies_hz, values_db)``.
    """
    R = np.asarray(table.impedances_ohm, dtype=float)
    f = np.asarray(table.frequencies_mhz, dtype=float) * 1e6
    g = np.asarray(table.values_db, dtype=float)
    return R, f, g


def _model_dips(f_grid, R, cable, bridge, f_meas, depth_meas, depth_weight):
    """Residual vector for one trial bridge: relative frequency error, plus depth if weighted."""
    sweep = resonance_sweep(f_grid, R, cable, bridge)
    res = []
    for i in range(len(R)):
        fk = np.atleast_1d(np.asarray(sweep.resonance_frequencies[i], dtype=float))
        gk = np.atleast_1d(np.asarray(sweep.gamma_at_resonance_db[i], dtype=float))
        if fk.size == 0 or not np.all(np.isfinite(fk)):
            res.append(NO_DIP_PENALTY)
            if depth_weight:
                res.append(NO_DIP_PENALTY)
            continue
        j = int(np.argmin(np.abs(fk - f_meas[i])))
        res.append((fk[j] - f_meas[i]) / f_meas[i])
        if depth_weight:
            res.append(depth_weight * (gk[j] - depth_meas[i]))
    return np.asarray(res)


def model_frequencies(resistances, measured_frequencies_hz, cable=CableParams(),
                      bridge=BridgeParams(), f=None):
    """
    Model dip frequency nearest each measured frequency, one per resistance.

    Entries are NaN where the model has no dip at that resistance.
    """
    if f is None:
        f = np.geomspace(5e6, 300e6, 4000)
    R = np.asarray(resistances, dtype=float)
    sweep = resonance_sweep(f, R, cable, bridge)
    out = np.full(len(R), np.nan)
    for i in range(len(R)):
        fk = np.atleast_1d(np.asarray(sweep.resonance_frequencies[i], dtype=float))
        if fk.size and np.all(np.isfinite(fk)):
            out[i] = fk[np.argmin(np.abs(fk - measured_frequencies_hz[i]))]
    return out


def fit_bridge(resistances, measured_frequencies_hz, measured_db=None, *,
               fit_params: Sequence[str] = DEFAULT_FIT_PARAMS,
               start: BridgeParams = BridgeParams(),
               cable: CableParams = CableParams(),
               f=None, depth_weight: float = 0.0, log_span: float = 1.5,
               max_nfev: Optional[int] = None) -> FitResult:
    """
    Fit selected bridge components to measured resonance frequencies.

    Parameters
    ----------
    resistances : (n,) array
        Load resistance for each measurement (ohm).
    measured_frequencies_hz : (n,) array
        Measured dip frequency for each resistance (Hz).
    measured_db : (n,) array, optional
        Measured dip depth (dB). Used only when ``depth_weight`` > 0.
    fit_params : names of ``BridgeParams`` fields to vary. Others stay at ``start``.
    start : BridgeParams
        Starting point, and the values of any parameters not being fitted.
    cable : CableParams
        Fixed cable model.
    f : (N,) array, optional
        Frequency grid for the model. Defaults to a log grid from 5 MHz to 300 MHz.
    depth_weight : float
        Weight on the depth residual, in units of 1/dB relative to the frequency
        residual. The default 0 fits frequencies only.
    log_span : float
        Each parameter may move by up to this many decades either side of its
        starting value.
    max_nfev : int, optional
        Limit on function evaluations, passed to scipy.

    Returns
    -------
    FitResult
    """
    R = np.atleast_1d(np.asarray(resistances, dtype=float))
    f_meas = np.atleast_1d(np.asarray(measured_frequencies_hz, dtype=float))
    if len(R) != len(f_meas):
        raise ValueError("resistances and measured_frequencies_hz must have the same length")
    if depth_weight:
        if measured_db is None:
            raise ValueError("depth_weight > 0 requires measured_db")
        g_meas = np.asarray(measured_db, dtype=float)
    else:
        g_meas = None

    names = tuple(fit_params)
    unknown = set(names) - {fld for fld in BridgeParams.__dataclass_fields__}
    if unknown:
        raise KeyError(f"Unknown bridge parameter(s): {sorted(unknown)}")
    x0 = np.array([np.log10(getattr(start, n)) for n in names])
    if not np.all(np.isfinite(x0)):
        raise ValueError("Starting values must be positive and finite")
    lo, hi = x0 - log_span, x0 + log_span

    if f is None:
        f = np.geomspace(5e6, 300e6, 4000)
    f = np.asarray(f, dtype=float)

    def fun(x):
        bridge = replace(start, **{n: 10 ** v for n, v in zip(names, x)})
        return _model_dips(f, R, cable, bridge, f_meas, g_meas, depth_weight)

    kwargs = {} if max_nfev is None else {"max_nfev": max_nfev}
    sol = least_squares(fun, x0, bounds=(lo, hi), **kwargs)

    fitted = {n: 10 ** v for n, v in zip(names, sol.x)}
    bridge = replace(start, **fitted)

    # Approximate 1-sigma errors from the Jacobian, in log10 space, then converted
    # to SI (d value = value * ln10 * d log10 value).
    m, n = sol.fun.size, sol.x.size
    stderr = {k: np.nan for k in names}
    if m > n:
        s2 = 2 * sol.cost / (m - n)
        try:
            cov = np.linalg.pinv(sol.jac.T @ sol.jac) * s2
            for i, k in enumerate(names):
                if cov[i, i] >= 0:
                    stderr[k] = fitted[k] * np.log(10) * np.sqrt(cov[i, i])
        except np.linalg.LinAlgError:
            pass

    model_f = model_frequencies(R, f_meas, cable, bridge, f=f)
    finite = np.isfinite(model_f)
    rel = (model_f[finite] - f_meas[finite]) / f_meas[finite]
    rms = float(np.sqrt(np.mean(rel ** 2))) if rel.size else np.nan

    return FitResult(
        bridge=bridge,
        params=fitted,
        stderr=stderr,
        resistances=R,
        measured_frequencies_hz=f_meas,
        model_frequencies_hz=model_f,
        residuals=sol.fun,
        rms_relative_error=rms,
        cost=float(sol.cost),
        success=bool(sol.success),
        message=str(sol.message),
        nfev=int(sol.nfev),
        fit_params=names,
    )


def fit_bridge_multistart(resistances, measured_frequencies_hz, measured_db=None, *,
                          scales=(0.5, 1.0, 2.0), start: BridgeParams = BridgeParams(),
                          **kwargs):
    """
    Run ``fit_bridge`` from several starting points and return every result.

    Each start multiplies the reactive components of ``start`` by one of ``scales``.
    The fit is often ill-conditioned, so different starts can land on different
    solutions with similar residuals. The spread across the returned list is a
    rough indication of how well the parameters are determined.

    Returns the list of ``FitResult`` sorted by RMS relative error, best first.
    """
    fit_params = kwargs.pop("fit_params", DEFAULT_FIT_PARAMS)
    results = []
    for sc in scales:
        s0 = replace(start, **{n: getattr(start, n) * sc for n in fit_params})
        results.append(fit_bridge(resistances, measured_frequencies_hz, measured_db,
                                  fit_params=fit_params, start=s0, **kwargs))
    results.sort(key=lambda r: r.rms_relative_error if np.isfinite(r.rms_relative_error)
                 else np.inf)
    return results


@dataclass
class SpectrumFitResult:
    """Outcome of ``fit_spectra``: the bridge and cable fitted to full measured S11 traces."""

    bridge: BridgeParams
    cable: CableParams                    # cable used, with any fitted length
    params: Dict[str, float]              # fitted bridge values, SI units
    cable_params: Dict[str, float]        # fitted cable values, SI units
    stderr: Dict[str, float]              # approximate 1-sigma, SI units (nan if undefined)
    resistances: np.ndarray               # ohm, labelled value per trace
    rms_db: float                         # RMS of (measured - model) over all points, dB
    rms_db_per_trace: Dict[float, float]  # labelled resistance -> RMS dB for that trace
    fitted_resistances: np.ndarray        # ohm, as fitted (equal to labels if not fitted)
    resistance_at_bound: np.ndarray       # True where a fitted resistance hit its window edge
    residuals: np.ndarray
    cost: float
    success: bool
    message: str
    nfev: int = 0
    fit_params: Tuple[str, ...] = field(default=DEFAULT_FIT_PARAMS)
    freq_range: Optional[Tuple[float, float]] = None


def _spectrum_data(traces, freq_range):
    """(resistance, frequencies, measured dB) for each trace, cropped to ``freq_range``."""
    data = []
    for tr in traces:
        if not isinstance(tr, VnaTrace):
            raise TypeError("traces must be VnaTrace objects, e.g. from load_vna_csv")
        if tr.resistance_ohm is None:
            raise ValueError(f"Trace {tr.name!r} has no resistance; pass resistance_ohm to load_vna_csv")
        f, g = np.asarray(tr.freq_hz, float), np.asarray(tr.s11_db, float)
        if freq_range is not None:
            lo, hi = freq_range
            keep = (f >= lo) & (f <= hi)
            f, g = f[keep], g[keep]
        if f.size == 0:
            raise ValueError(f"No points of trace {tr.name!r} lie within freq_range")
        data.append((float(tr.resistance_ohm), f, g))
    return data


def spectrum_model_db(trace, bridge, cable=ZERO_CABLE, freq_range=None):
    """Theoretical S11 (dB) of ``bridge`` (behind ``cable``) at the trace's resistance and frequencies."""
    (R, f, _), = _spectrum_data([trace], freq_range)
    return reflection_spectrum_db(f, R, cable, bridge)


CABLE_FIT_PARAMS = ("length",)


def fit_spectra(traces, *, fit_params: Sequence[str] = DEFAULT_FIT_PARAMS,
                start: BridgeParams = BridgeParams(),
                cable: CableParams = ZERO_CABLE,
                fit_cable: Sequence[str] = (),
                cable_bounds: Tuple[float, float] = (0.0, 5.0),
                cable_scan_points: int = 61,
                freq_range: Optional[Tuple[float, float]] = None,
                log_span: float = 1.5, max_nfev: Optional[int] = None,
                fit_resistances: bool = False,
                resistance_tolerance: float = 0.1) -> SpectrumFitResult:
    """
    Fit selected bridge components (and optionally the cable length) so the
    theoretical S11 matches measured S11.

    Each trace is compared with the theoretical S11 at that trace's resistance,
    point by point in frequency. All traces share one set of bridge and cable
    values. The residual is measured minus model, in dB.

    Parameters
    ----------
    traces : list of VnaTrace
        Measured traces, each with ``resistance_ohm`` set.
    fit_params : names of ``BridgeParams`` fields to vary. Others stay at ``start``.
    start : BridgeParams
        Starting point, and the values of any bridge parameters not being fitted.
    cable : CableParams
        Base cable. Its length is replaced by the fitted value if ``fit_cable``
        includes ``"length"``; otherwise it is used as given.
    fit_cable : subset of ``("length",)``
        Cable parameters to fit. Length is fitted linearly within ``cable_bounds``
        (metres), after a coarse scan over the same range to pick the starting point.
    cable_bounds : (min, max) metres
        Allowed range for the fitted cable length.
    cable_scan_points : int
        Number of lengths in the coarse scan.
    freq_range : (f_low, f_high) in Hz, optional
        Fit only points inside this band.
    log_span : float
        Each bridge parameter may move by up to this many decades either side of its start.
    max_nfev : int, optional
        Limit on function evaluations, passed to scipy.
    fit_resistances : bool
        If True, each trace's load resistance is also a fit parameter.
    resistance_tolerance : float
        With ``fit_resistances``, each resistance may move within this fraction of
        its labelled value, e.g. 0.1 allows +/-10%.

    Returns
    -------
    SpectrumFitResult
    """
    data = _spectrum_data(traces, freq_range)
    labels = np.array([R for R, _, _ in data])

    names = tuple(fit_params)
    unknown = set(names) - set(BridgeParams.__dataclass_fields__)
    if unknown:
        raise KeyError(f"Unknown bridge parameter(s): {sorted(unknown)}")
    bad_cable = set(fit_cable) - set(CABLE_FIT_PARAMS)
    if bad_cable:
        raise KeyError(f"Unsupported cable parameter(s) to fit: {sorted(bad_cable)}; "
                       f"supported: {CABLE_FIT_PARAMS}")
    x0 = np.array([np.log10(getattr(start, n)) for n in names])
    if not np.all(np.isfinite(x0)):
        raise ValueError("Starting values must be positive and finite")
    lo, hi = x0 - log_span, x0 + log_span
    nb = len(names)

    # Index layout of the parameter vector:
    #   [log10 bridge values..., log10 resistances (if fitted)..., cable length (if fitted)]
    fit_len = "length" in fit_cable
    if fit_len:
        lmin, lmax = cable_bounds
        if not 0 <= lmin < lmax:
            raise ValueError("cable_bounds must satisfy 0 <= min < max")

    def model_residuals(bridge, Rs, cab):
        parts = []
        for R, (_, f, g) in zip(Rs, data):
            r = g - reflection_spectrum_db(f, R, cab, bridge)
            parts.append(np.where(np.isfinite(r), r, 100.0))   # guard against -inf models
        return np.concatenate(parts)

    # Coarse scan over cable length, bridge fixed at start, to choose the starting length.
    cab0 = cable
    if fit_len:
        scan = np.linspace(lmin, lmax, cable_scan_points)
        scores = [np.sum(model_residuals(start, labels, replace(cable, length=L)) ** 2) for L in scan]
        cab0 = replace(cable, length=float(scan[int(np.argmin(scores))]))

    if fit_resistances:
        if not 0 < resistance_tolerance < 1:
            raise ValueError("resistance_tolerance must be between 0 and 1")
        x0 = np.concatenate([x0, np.log10(labels)])
        lo = np.concatenate([lo, np.log10(labels * (1 - resistance_tolerance))])
        hi = np.concatenate([hi, np.log10(labels * (1 + resistance_tolerance))])
    if fit_len:
        x0 = np.concatenate([x0, [cab0.length]])
        lo = np.concatenate([lo, [lmin]])
        hi = np.concatenate([hi, [lmax]])

    nr = len(labels) if fit_resistances else 0

    def unpack(x):
        bridge = replace(start, **{n: 10 ** v for n, v in zip(names, x[:nb])})
        Rs = 10 ** x[nb:nb + nr] if fit_resistances else labels
        cab = replace(cab0, length=float(x[-1])) if fit_len else cab0
        return bridge, Rs, cab

    def fun(x):
        bridge, Rs, cab = unpack(x)
        return model_residuals(bridge, Rs, cab)

    kwargs = {} if max_nfev is None else {"max_nfev": max_nfev}
    sol = least_squares(fun, x0, bounds=(lo, hi), **kwargs)

    bridge, Rs, cab = unpack(sol.x)
    fitted = {n: 10 ** v for n, v in zip(names, sol.x[:nb])}
    cable_fitted = {"length": cab.length} if fit_len else {}
    at_bound = np.zeros(len(labels), dtype=bool)
    if fit_resistances:
        rx, rlo, rhi = sol.x[nb:nb + nr], lo[nb:nb + nr], hi[nb:nb + nr]
        at_bound = np.isclose(rx, rlo, atol=1e-6) | np.isclose(rx, rhi, atol=1e-6)

    m, n = sol.fun.size, sol.x.size
    stderr = {k: np.nan for k in names}
    if fit_len:
        stderr["length"] = np.nan
    if m > n:
        s2 = 2 * sol.cost / (m - n)
        try:
            cov = np.linalg.pinv(sol.jac.T @ sol.jac) * s2
            for i, k in enumerate(names):
                if cov[i, i] >= 0:
                    stderr[k] = fitted[k] * np.log(10) * np.sqrt(cov[i, i])
            if fit_len and cov[-1, -1] >= 0:
                stderr["length"] = np.sqrt(cov[-1, -1])
        except np.linalg.LinAlgError:
            pass

    per_trace, all_sq = {}, []
    for lab, R, (_, f, g) in zip(labels, Rs, data):
        r = g - reflection_spectrum_db(f, R, cab, bridge)
        per_trace[float(lab)] = float(np.sqrt(np.mean(r ** 2)))
        all_sq.append(r ** 2)
    rms = float(np.sqrt(np.mean(np.concatenate(all_sq))))

    return SpectrumFitResult(
        bridge=bridge,
        cable=cab,
        params=fitted,
        cable_params=cable_fitted,
        stderr=stderr,
        resistances=labels,
        rms_db=rms,
        rms_db_per_trace=per_trace,
        fitted_resistances=np.asarray(Rs, dtype=float),
        resistance_at_bound=at_bound,
        residuals=sol.fun,
        cost=float(sol.cost),
        success=bool(sol.success),
        message=str(sol.message),
        nfev=int(sol.nfev),
        fit_params=names,
        freq_range=freq_range,
    )


def fit_spectra_multistart(traces, *, scales=(0.25, 0.5, 1.0, 2.0, 4.0),
                           start: BridgeParams = BridgeParams(), **kwargs):
    """
    Run ``fit_spectra`` from several starting points; return all results, best first.

    Each start multiplies the varied bridge components of ``start`` by one of ``scales``.
    """
    fit_params = kwargs.pop("fit_params", DEFAULT_FIT_PARAMS)
    results = []
    for sc in scales:
        s0 = replace(start, **{n: getattr(start, n) * sc for n in fit_params})
        results.append(fit_spectra(traces, fit_params=fit_params, start=s0, **kwargs))
    results.sort(key=lambda r: r.rms_db if np.isfinite(r.rms_db) else np.inf)
    return results


# ---------------------------------------------------------------------------
# Short-circuit cable fit: far end of the cable connected straight to ground.
# ---------------------------------------------------------------------------

CABLE_SHORT_PARAMS = ("length", "loss_sqrt", "loss_lin")

# Allowed ranges: length in metres, loss coefficients non-negative.
CABLE_SHORT_BOUNDS = {"length": (0.0, 5.0), "loss_sqrt": (0.0, 20.0), "loss_lin": (0.0, 1.0)}


@dataclass
class CableFitResult:
    """Outcome of ``fit_short_cable``."""

    cable: CableParams                    # fitted cable (other fields from ``start``)
    params: Dict[str, float]              # fitted values of the varied fields
    stderr: Dict[str, float]              # approximate 1-sigma (nan if undefined)
    rms_db: float                         # RMS of (measured - model) over the fitted band, dB
    residuals: np.ndarray
    cost: float
    success: bool
    message: str
    nfev: int = 0
    freq_range: Optional[Tuple[float, float]] = None


def short_cable_model_db(f, cable):
    """Theoretical S11 (dB) of a cable whose far end is shorted to ground."""
    abcd, _, _ = cable_model(np.asarray(f, dtype=float), cable)
    Zin = abcd[0, 1] / abcd[1, 1]        # Z_in = B / D when the load is 0 ohm
    return reflection_db(Zin, cable.Z0c)


def fit_short_cable(freq_hz, s11_db, *, start: CableParams = CableParams(length=1.0),
                    fit: Sequence[str] = CABLE_SHORT_PARAMS,
                    freq_range: Optional[Tuple[float, float]] = None,
                    global_search: bool = True, seed: int = 0,
                    scan_points: int = 2501, n_candidates: int = 10,
                    max_nfev: Optional[int] = None) -> CableFitResult:
    """
    Fit cable properties to a measured short-circuit S11 trace.

    Parameters
    ----------
    freq_hz, s11_db : (N,) arrays
        Measured frequency (Hz) and S11 (dB) with the cable's far end shorted.
    start : CableParams
        Base cable. Fields not in ``fit`` keep these values.
    fit : subset of ``CABLE_SHORT_PARAMS``
        Which fields to vary: ``length`` (m), ``loss_sqrt``, ``loss_lin``.
    freq_range : (f_low, f_high) in Hz, optional
        Fit only points inside this band.
    global_search : bool
        Run a differential-evolution search over the fitted fields before the local
        polish. The length and loss coefficients are coupled, so a local search from
        one start can settle in the wrong basin.
    seed : int
        Random seed for the global search, so results are reproducible.
    scan_points : int
        Points in the scan over cable length. The S11 oscillates rapidly with
        length (about every 2 cm at 6 GHz), so the scan step must be well below that;
        the default gives a step of 2 mm over 0-5 m.
    n_candidates : int
        Number of best scan minima to refine from. The best refined result wins.

    Notes
    -----
    The phase depends only on length / VF, and the characteristic impedance is
    fixed by the reference impedance, so length and velocity factor cannot be
    separated. The fit keeps VF at its value in ``start``; the fitted length is
    the electrical length for that VF.
    """
    f = np.asarray(freq_hz, dtype=float)
    g = np.asarray(s11_db, dtype=float)
    if freq_range is not None:
        keep = (f >= freq_range[0]) & (f <= freq_range[1])
        f, g = f[keep], g[keep]
    if f.size < 4:
        raise ValueError("Need at least 4 points inside the fit band")

    fit = tuple(fit)
    bad = set(fit) - set(CABLE_SHORT_PARAMS)
    if bad:
        raise KeyError(f"Unsupported cable parameter(s): {sorted(bad)}; supported: {CABLE_SHORT_PARAMS}")

    def residual(cab):
        r = g - short_cable_model_db(f, cab)
        return np.where(np.isfinite(r), r, 100.0)

    def with_values(x):
        return replace(start, **{n: float(v) for n, v in zip(fit, x)})

    # Start values for the fit: length from a coarse scan (other fields from start).
    lo = np.array([CABLE_SHORT_BOUNDS[n][0] for n in fit])
    hi = np.array([CABLE_SHORT_BOUNDS[n][1] for n in fit])
    x_start = np.array([np.clip(getattr(start, n), l, h) for n, l, h in zip(fit, lo, hi)])

    candidates = [x_start]
    if "length" in fit:
        i_len = fit.index("length")
        grid = np.linspace(lo[i_len], hi[i_len], scan_points)
        scores = []
        for Lg in grid:
            xg = x_start.copy(); xg[i_len] = Lg
            scores.append(np.sum(residual(with_values(xg)) ** 2))
        scores = np.asarray(scores)
        # local minima of the scan, best first
        mins = [i for i in range(1, len(scores) - 1)
                if scores[i] <= scores[i - 1] and scores[i] <= scores[i + 1]]
        mins.sort(key=lambda i: scores[i])
        candidates = []
        for i in mins[:n_candidates]:
            xg = x_start.copy(); xg[i_len] = grid[i]
            candidates.append(xg)
        if not candidates:
            candidates = [x_start]

    kwargs = {} if max_nfev is None else {"max_nfev": max_nfev}
    if global_search:
        de = differential_evolution(lambda x: float(np.sum(residual(with_values(x)) ** 2)),
                                    bounds=list(zip(lo, hi)), seed=seed, maxiter=400,
                                    tol=1e-10, polish=False, init="sobol")
        candidates = [de.x] + candidates
    best = None
    for x0 in candidates:
        sol = least_squares(lambda x: residual(with_values(x)), x0,
                            bounds=(lo, hi), x_scale="jac", **kwargs)
        if best is None or sol.cost < best.cost:
            best = sol

    cab = with_values(best.x)
    m, n = best.fun.size, best.x.size
    stderr = {k: np.nan for k in fit}
    if m > n:
        s2 = 2 * best.cost / (m - n)
        try:
            cov = np.linalg.pinv(best.jac.T @ best.jac) * s2
            for i, k in enumerate(fit):
                if cov[i, i] >= 0:
                    stderr[k] = float(np.sqrt(cov[i, i]))
        except np.linalg.LinAlgError:
            pass

    r = g - short_cable_model_db(f, cab)
    rms = float(np.sqrt(np.mean(r[np.isfinite(r)] ** 2)))

    return CableFitResult(
        cable=cab,
        params={k: float(getattr(cab, k)) for k in fit},
        stderr=stderr,
        rms_db=rms,
        residuals=best.fun,
        cost=float(best.cost),
        success=bool(best.success),
        message=str(best.message),
        nfev=int(best.nfev),
        freq_range=freq_range,
    )
