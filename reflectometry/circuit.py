"""Lumped bridge circuit expressed as cascaded 2x2 (ABCD-style) matrices."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BridgeParams:
    """
    Component values of the bridge load. Defaults match the original notebook.

    ``R_eq`` is the termination used when converting the network into a load
    impedance for plotting (see ``plotting.plot_cable_effect``).
    """

    R_T: float = 4700.0       # ohm
    C_T: float = 56e-12       # F
    L: float = 470e-9         # H
    R_L: float = 1.1         # ohm
    C_P: float = 6e-12        # F
    L_P: float = 2e-6           # H
    R_eq: float = 30000.0     # ohm


def cascade(*mats):
    """
    Multiply (2, 2, N) matrix stacks along the frequency axis.

    Each frequency point is multiplied independently: the result at index n is
    ``mats[0][:, :, n] @ mats[1][:, :, n] @ ...``.
    """
    out = mats[0]
    for m in mats[1:]:
        out = np.einsum("abn,bcn->acn", out, m)
    return out


def bridge_matrix(f, bridge=BridgeParams()):
    """
    Cascaded matrix of the bridge network at frequencies ``f`` (Hz).

    Returns an array of shape (2, 2, N).
    """
    f = np.asarray(f, dtype=float)
    n = len(f)
    omega = 2 * np.pi * f

    M1 = np.array([[np.ones(n), np.zeros(n)],
                   [np.full(n, 1 / bridge.R_T), np.ones(n)]])
    M2 = np.array([[np.ones(n), 1 / (1j * omega * bridge.C_T)],
                   [np.zeros(n), np.ones(n)]])
    M3 = np.array([[np.ones(n), 1j * omega * bridge.L],
                   [np.zeros(n), np.ones(n)]])
    M4 = np.array([[np.ones(n), np.full(n, bridge.R_L)],
                   [np.zeros(n), np.ones(n)]])
    M5 = np.array([[np.ones(n), np.zeros(n)],
                   [1j * omega * bridge.C_P, np.ones(n)]])
    M6 = np.array([[np.ones(n), 1j * omega * bridge.L_P],
                   [np.zeros(n), np.ones(n)]])

    return cascade(M1, M2, M3, M4, M5, M6)
