"""Loading of VNA traces exported as delimited text (e.g. ``3675Ohm.csv``)."""

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .reflection import minima_from_db

_FREQ_UNITS = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
_FREQ_COL_RE = re.compile(r"freq\[(\w+)\]", re.IGNORECASE)
# "_" may stand in for the decimal point in filenames, e.g. 0_35Ohm.csv -> 0.35 ohm
_RESISTANCE_RE = re.compile(r"(\d+(?:[._]\d+)?)\s*ohm", re.IGNORECASE)


@dataclass(frozen=True)
class VnaTrace:
    """One S11 trace, sorted by frequency."""

    freq_hz: np.ndarray
    s11_db: np.ndarray
    name: str
    resistance_ohm: Optional[float] = None
    header: Tuple[str, ...] = ()        # comment lines from the file

    def minimum(self):
        """(frequency in Hz, value in dB) of the deepest sample."""
        i = int(np.argmin(self.s11_db))
        return self.freq_hz[i], self.s11_db[i]


def load_vna_csv(path, resistance_ohm=None):
    """
    Read a VNA trace exported as text: optional ``#`` comment lines, an optional
    column-name line, then rows of frequency and S11 in dB (tab or comma separated).

    The frequency unit is taken from the column name (e.g. ``freq[MHz]``) and
    converted to Hz; Hz is assumed if there is no unit. If ``resistance_ohm`` is
    not given, it is read from the filename (e.g. ``3675Ohm.csv`` -> 3675).
    """
    path = os.fspath(path)
    name = os.path.splitext(os.path.basename(path))[0]
    header, columns, rows = [], None, []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                text = line.lstrip("#").strip()
                if text:
                    header.append(text)
                continue
            sep = "\t" if "\t" in line else ","
            parts = [p.strip() for p in line.split(sep)]
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                if not rows and columns is None:
                    columns = parts        # the column-name line, before any data
                continue

    if not rows:
        raise ValueError(f"No numeric data found in {path}")

    scale = 1.0
    if columns:
        m = _FREQ_COL_RE.search(columns[0])
        if m and m.group(1).lower() in _FREQ_UNITS:
            scale = _FREQ_UNITS[m.group(1).lower()]

    data = np.array(rows)
    order = np.argsort(data[:, 0])
    if resistance_ohm is None:
        m = _RESISTANCE_RE.search(name)
        resistance_ohm = float(m.group(1).replace("_", ".")) if m else None

    return VnaTrace(
        freq_hz=data[order, 0] * scale,
        s11_db=data[order, 1],
        name=name,
        resistance_ohm=resistance_ohm,
        header=tuple(header),
    )


def find_trace_minima(trace, min_depth_db=None):
    """
    Interpolated dips of a VNA trace, found with the same method as the model.

    Returns ``(freq_hz, s11_db)`` arrays sorted by frequency. Only interior local
    minima are returned. With ``min_depth_db``, only dips at or below that level
    are kept, e.g. ``-3.0`` keeps dips deeper than 3 dB.
    """
    f, d = minima_from_db(trace.freq_hz, trace.s11_db)[0]
    keep = np.ones(len(f), dtype=bool) if min_depth_db is None else d <= min_depth_db
    order = np.argsort(f[keep])
    return f[keep][order], d[keep][order]
