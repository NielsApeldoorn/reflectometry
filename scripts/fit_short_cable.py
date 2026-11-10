"""
Fit cable length and loss coefficients from a shorted-cable VNA trace.

Example:
    python scripts/fit_short_cable.py data/cable.csv --band 0.1 200 --plot cable_fit.png

--band takes a frequency range in MHz. Omit it to fit the full trace.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from reflectometry import CableParams, fit_short_cable, load_vna_csv, short_cable_model_db  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="shorted-cable VNA export")
    ap.add_argument("--band", nargs=2, type=float, metavar=("LO_MHZ", "HI_MHZ"),
                    help="fit only this frequency range (MHz)")
    ap.add_argument("--length-start", type=float, default=1.0, help="starting length (m)")
    ap.add_argument("--plot", metavar="PNG", help="save a measured-vs-fitted plot")
    args = ap.parse_args(argv)

    tr = load_vna_csv(args.csv, resistance_ohm=0.0)
    band = None if args.band is None else (args.band[0] * 1e6, args.band[1] * 1e6)
    res = fit_short_cable(tr.freq_hz, tr.s11_db, start=CableParams(length=args.length_start),
                          freq_range=band)

    where = "full trace" if band is None else f"{args.band[0]:g}-{args.band[1]:g} MHz"
    print(f"{os.path.basename(args.csv)}, {where}: RMS {res.rms_db:.4f} dB, success={res.success}")
    for k in ("length", "loss_sqrt", "loss_lin"):
        print(f"  {k:10s} = {res.params[k]:.6g}  (1-sigma {res.stderr[k]:.3g})")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m = np.ones_like(tr.freq_hz, bool) if band is None else \
            (tr.freq_hz >= band[0]) & (tr.freq_hz <= band[1])
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(tr.freq_hz[m] / 1e6, tr.s11_db[m], lw=0.8, label="measured (shorted)")
        ax.plot(tr.freq_hz[m] / 1e6, short_cable_model_db(tr.freq_hz[m], res.cable), lw=0.8, ls="--",
                label=f"fit: L={res.params['length']:.3f} m")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("S11 (dB)")
        ax.set_title(f"Short-circuit cable, {where}")
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"plot saved to {args.plot}")
    return res


if __name__ == "__main__":
    main()
