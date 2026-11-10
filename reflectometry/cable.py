"""Transmission-line model of the coaxial cable."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CableParams:
    """
    Physical parameters of the cable.

    The defaults reproduce the original notebook. ``loss_sqrt`` and ``loss_lin``
    are the datasheet loss coefficients: the attenuation is
    ``loss_sqrt * sqrt(f_MHz) + loss_lin * f_MHz``.
    """

    length: float = 2.0          # m
    Z0c: float = 50.0            # reference / characteristic impedance (ohm)
    VF: float = 0.7              # velocity factor
    c: float = 3e8               # speed of light (m/s)
    loss_sqrt: float = 2.583794  # datasheet coefficient of sqrt(f in MHz)
    loss_lin: float = 0.003893   # datasheet coefficient of f in MHz


def cable_model(f, cable=CableParams()):
    """
    ABCD model of the cable over a frequency grid.

    Parameters
    ----------
    f : array_like
        Frequencies in Hz.
    cable : CableParams

    Returns
    -------
    abcd : ndarray, shape (2, 2, N)
        ABCD matrix at each frequency.
    Zc : ndarray, shape (N,)
        Complex characteristic impedance (ohm).
    gamma : ndarray, shape (N,)
        Complex propagation constant (1/m).
    """
    f = np.asarray(f, dtype=float)
    omega = 2 * np.pi * f

    Z0c, VF, c, length = cable.Z0c, cable.VF, cable.c, cable.length
    Lp = Z0c / (VF * c)            # H/m
    Cp = 1 / (Z0c * VF * c)        # F/m

    alpha = cable.loss_sqrt * np.sqrt(f * 1e-6) + cable.loss_lin * f * 1e-6
    alpha = alpha / 100 / 8.686    # datasheet units -> Np/m
    Rp = 2 * Z0c * alpha           # ohm/m (simplified)

    Zc = np.sqrt((Rp + 1j * omega * Lp) / (1j * omega * Cp))
    gamma = np.sqrt((Rp + 1j * omega * Lp) * (1j * omega * Cp))

    ch = np.cosh(gamma * length)
    sh = np.sinh(gamma * length)
    abcd = np.array([[ch, Zc * sh],
                     [sh / Zc, ch]])
    return abcd, Zc, gamma
