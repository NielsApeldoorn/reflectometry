"""Quadrature (I/Q) voltages for a series RLC load fed from a source impedance."""

import numpy as np


def resonant_frequency(L, C):
    """Series resonance frequency in Hz of an LC pair."""
    return 1 / np.sqrt(L * C) / (2 * np.pi)


def series_rlc_impedance(omega, L, C, R):
    """Impedance of a series R, L, C branch at angular frequency ``omega``."""
    return 1j * omega * L + 1 / (1j * omega * C) + R


def quadratures(Z_in, Z_load, V_in=1, gain=0):
    """
    In-phase and quadrature voltages across ``Z_load`` driven from ``Z_in``.

    Parameters
    ----------
    Z_in : complex
        Source impedance (ohm).
    Z_load : complex or array
        Load impedance (ohm).
    V_in : float
        Source amplitude (V).
    gain : float
        Gain in dB applied to the output.

    Returns
    -------
    V_I, V_Q : real arrays (or scalars)
    """
    V = 10 ** (gain / 20) * V_in * Z_load / (Z_load + Z_in)
    return np.real(V), np.imag(V)
