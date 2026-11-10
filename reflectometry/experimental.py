"""Measured data from the bench, for overlay on model results."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class MeasuredResonances:
    """Measured dip frequency and depth for a set of load resistances."""

    impedances_ohm: Tuple[float, ...]
    frequencies_mhz: Tuple[float, ...]
    values_db: Tuple[float, ...]

    def sorted_by_impedance(self):
        """Return ``(z, f_mhz, db)`` as tuples, sorted by impedance for line plots."""
        data = sorted(zip(self.impedances_ohm, self.frequencies_mhz, self.values_db))
        z, f, db = zip(*data)
        return np.array(z), np.array(f), np.array(db)


# Resonator from the bench: inductance and effective capacitance (used in the
# "resonance at zero" check and when comparing against the model)
MEASURED_INDUCTANCE_H = 0.47e-6
MEASURED_CAPACITANCE_F = 8.420976034103871e-12

MEASURED_RESONANCES = MeasuredResonances(
    impedances_ohm=(75, 180, 700, 1460, 1767.5, 2850, 6157, 4768, 13250, 20400, 36480),
    frequencies_mhz=(26.44, 41.35, 71.87, 76.34, 77.30, 78.25, 78.49, 78.72,
                     79.21, 79.48, 79.6),
    values_db=(-10.3, -6.0, -10.4, -21.1, -31.8, -19.1, -11.5, -12.3,
               -8.6, -7.5, -7.7),
)
