"""VNA CSV loading and trace-minimum finding."""
import os
import tempfile

import numpy as np

from reflectometry.vna import find_trace_minima, load_vna_csv


def _write(directory, name, rows, unit="freq[Hz]", crlf=True):
    path = os.path.join(directory, name)
    nl = "\r\n" if crlf else "\n"
    lines = ["# Version 1.00", "#", f"{unit}\tTrc2_S11[dB]\t"]
    lines += [f"{f:.15E}\t{g:.15E}\t" for f, g in rows]
    with open(path, "w", newline="") as fh:
        fh.write(nl.join(lines) + nl)
    return path


def test_load_reads_frequency_and_s11():
    with tempfile.TemporaryDirectory() as d:
        rows = [(1e5, -0.1), (2e5, -0.2), (3e5, -0.3)]
        tr = load_vna_csv(_write(d, "3675Ohm.csv", rows))
        assert np.allclose(tr.freq_hz, [1e5, 2e5, 3e5])
        assert np.allclose(tr.s11_db, [-0.1, -0.2, -0.3])
        assert tr.resistance_ohm == 3675.0
        assert tr.header == ("Version 1.00",)


def test_load_scales_mhz_column():
    with tempfile.TemporaryDirectory() as d:
        rows = [(0.1, -0.1), (0.2, -0.2)]
        tr = load_vna_csv(_write(d, "x.csv", rows, unit="freq[MHz]", crlf=False))
        assert np.allclose(tr.freq_hz, [1e5, 2e5])


def test_decimal_underscore_in_filename():
    with tempfile.TemporaryDirectory() as d:
        tr = load_vna_csv(_write(d, "0_35Ohm.csv", [(1e5, -0.1), (2e5, -0.2)]))
        assert tr.resistance_ohm == 0.35


def test_suffix_after_resistance_is_ignored():
    with tempfile.TemporaryDirectory() as d:
        tr = load_vna_csv(_write(d, "3675Ohm_1.csv", [(1e5, -0.1), (2e5, -0.2)]))
        assert tr.resistance_ohm == 3675.0


def test_unsorted_rows_are_sorted():
    with tempfile.TemporaryDirectory() as d:
        tr = load_vna_csv(_write(d, "y.csv", [(3e5, -0.3), (1e5, -0.1), (2e5, -0.2)]))
        assert np.all(np.diff(tr.freq_hz) > 0)


def test_find_trace_minima_recovers_dip_and_filters_by_depth():
    f = np.linspace(1e6, 2e8, 2000)
    shallow = -1.0 + 0.0 * f
    dB = shallow + (-6.0 + 1.0) * np.exp(-((f - 47.3e6) / 2e6) ** 2) \
                 + (-2.0 + 1.0) * np.exp(-((f - 150.8e6) / 2e6) ** 2)
    from reflectometry.vna import VnaTrace
    tr = VnaTrace(freq_hz=f, s11_db=dB, name="syn", resistance_ohm=1000.0)
    fd, dd = find_trace_minima(tr, min_depth_db=-3.0)
    assert len(fd) == 1
    assert abs(fd[0] - 47.3e6) < 1e4
