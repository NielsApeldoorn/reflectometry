"""
Fit bridge components (and optionally the cable length) to measured S11 traces.

Traces are loaded from --data-dir. The load resistance comes from each file name
(e.g. 2144Ohm_1.csv -> 2144 ohm). Components not listed in --fit stay at their
nominal values from BridgeParams.

Example:
    python scripts/fit_bridge_spectra.py --data-dir data \\
        --files 1147Ohm_1 2144Ohm_1 3675Ohm_cb7864 --fit C_P L_P R_L --fit-cable-length \\
        --plot bridge_fit.png
"""
import argparse
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reflectometry import (BridgeParams, CABLE_FIT_PARAMS, ZERO_CABLE,  # noqa: E402
                           fit_spectra_multistart, load_vna_csv)
from reflectometry.fitting import DEFAULT_FIT_PARAMS  # noqa: E402

DEFAULT_FILES = ["1147Ohm_1", "2144Ohm_1", "2684Ohm_1", "3223Ohm_1", "3497Ohm_1",
                 "3675Ohm_cb7864", "3749Ohm_1", "3811Ohm_1", "3950Ohm_1", "4071Ohm_1",
                 "4200Ohm_1", "4312Ohm_1"]


def load_traces(data_dir, names):
    """Load named traces, skipping byte-identical duplicate files."""
    seen, traces = set(), []
    for name in names:
        path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        digest = hashlib.md5(open(path, "rb").read()).hexdigest()
        if digest in seen:
            print(f"skipping duplicate of an earlier file: {name}.csv")
            continue
        seen.add(digest)
        traces.append(load_vna_csv(path))
    return sorted(traces, key=lambda t: t.resistance_ohm)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="directory containing the VNA CSV files")
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                    help="file names without .csv (default: the 12 traces used in the last fit)")
    ap.add_argument("--fit", nargs="+", default=list(("C_P", "L_P", "R_L")),
                    help="bridge components to vary (others stay nominal)")
    ap.add_argument("--fit-cable-length", action="store_true", help="also fit the cable length")
    ap.add_argument("--cable-bounds", nargs=2, type=float, default=(0.0, 5.0), metavar=("MIN_M", "MAX_M"))
    ap.add_argument("--freq-range", nargs=2, type=float, metavar=("LO_MHZ", "HI_MHZ"),
                    help="fit only this band (MHz)")
    ap.add_argument("--scales", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0],
                    help="multistart scale factors applied to the varied components")
    ap.add_argument("--max-nfev", type=int, default=3000)
    ap.add_argument("--plot", metavar="PNG", help="save the overlay of measured and fitted traces")
    args = ap.parse_args(argv)

    unknown = set(args.fit) - set(BridgeParams.__dataclass_fields__)
    if unknown:
        ap.error(f"unknown bridge parameter(s): {sorted(unknown)}")

    traces = load_traces(args.data_dir, args.files)
    print("traces:", [(t.name, t.resistance_ohm) for t in traces])

    band = None if args.freq_range is None else (args.freq_range[0] * 1e6, args.freq_range[1] * 1e6)
    fit_cable = CABLE_FIT_PARAMS if args.fit_cable_length else ()
    results = fit_spectra_multistart(
        traces, fit_params=tuple(args.fit), start=BridgeParams(), cable=ZERO_CABLE,
        fit_cable=fit_cable, cable_bounds=tuple(args.cable_bounds), freq_range=band,
        scales=tuple(args.scales), max_nfev=args.max_nfev)

    print("\nstarts (best first), RMS dB:", [round(r.rms_db, 4) for r in results])
    best = results[0]
    print(f"best: RMS {best.rms_db:.4f} dB, success={best.success}")
    for k in best.fit_params:
        print(f"  {k:6s} = {best.params[k]:.5g}  (1-sigma {best.stderr[k]:.3g})")
    if best.cable_params:
        print(f"  cable length = {best.cable.length:.4f} m  (1-sigma {best.stderr['length']:.3g})")
    print("per-trace RMS dB:", {k: round(v, 3) for k, v in best.rms_db_per_trace.items()})

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        from reflectometry.plotting import plot_fit_on_traces
        fig = plot_fit_on_traces(traces, best)
        fig.savefig(args.plot, dpi=120)
        print(f"plot saved to {args.plot}")
    return results


if __name__ == "__main__":
    main()
