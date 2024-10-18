from __future__ import annotations

import itertools
from itertools import chain, product

import numpy as np
import qutip as qt

from .utils.file_io import Serializable


class Model(Serializable):
    """Specify the model, including the Hamiltonian, drive strengths and frequencies.

    Can be subclassed to e.g. override the hamiltonian() method for a different (but
    still periodic!) Hamiltonian.

    Parameters:
        H0: Drift Hamiltonian, which must be diagonal and provided in units such that
            H0 can be passed directly to qutip.
        H1: Drive operator, which should be unitless (for instance the charge-number
            operator n of the transmon). It will be multiplied by a drive amplitude
            that we scan over from drive_parameters.drive_amplitudes.
        omega_d_values: drive frequencies to scan over
        drive_amplitudes: amp values to scan over. Can be one dimensional in which case
            these amplitudes are used for all omega_d, or it can be two dimensional
            in which case the first dimension are the amplitudes to scan over
            and the second are the amplitudes for respective drive frequencies

    """

    def __init__(
        self,
        H0: qt.Qobj | np.ndarray | list,
        H1: qt.Qobj | np.ndarray | list,
        omega_d_values: np.ndarray,
        drive_amplitudes: np.ndarray,
    ):
        if not isinstance(H0, qt.Qobj):
            H0 = qt.Qobj(np.array(H0, dtype=complex))
        if not isinstance(H1, qt.Qobj):
            H1 = qt.Qobj(np.array(H1, dtype=complex))
        if isinstance(omega_d_values, list):
            omega_d_values = np.array(omega_d_values)
        if isinstance(drive_amplitudes, list):
            drive_amplitudes = np.array(drive_amplitudes)
        if len(drive_amplitudes.shape) == 1:
            drive_amplitudes = np.tile(drive_amplitudes, (len(omega_d_values), 1)).T
        assert len(drive_amplitudes.shape) == 2
        assert drive_amplitudes.shape[1] == len(omega_d_values)

        self.H0 = H0
        self.H1 = H1
        self.omega_d_values = omega_d_values
        self.drive_amplitudes = drive_amplitudes

    def omega_d_to_idx(self, omega_d: float) -> np.ndarray[int]:
        """Return index corresponding to omega_d value."""
        return np.argmin(np.abs(self.omega_d_values - omega_d))

    def amp_to_idx(self, amp: float, omega_d: float) -> np.ndarray[int]:
        """Return index corresponding to amplitude value.

        Because the drive amplitude can depend on the drive frequency, we also must pass
        the drive frequency here.
        """
        omega_d_idx = self.omega_d_to_idx(omega_d)
        return np.argmin(np.abs(self.drive_amplitudes[:, omega_d_idx] - amp))

    def omega_d_amp_params(self, amp_idxs: list) -> itertools.chain:
        """Return ordered chain object of the specified omega_d and amplitude values."""
        amp_range_vals = self.drive_amplitudes[amp_idxs[0] : amp_idxs[1]]
        _omega_d_amp_params = [
            product([omega_d], amp_vals)
            for omega_d, amp_vals in zip(
                self.omega_d_values, amp_range_vals.T, strict=False
            )
        ]
        return chain(*_omega_d_amp_params)

    def hamiltonian(self, omega_d_amp: tuple[float, float]) -> list[qt.Qobj]:
        """Return the Hamiltonian we actually simulate."""
        omega_d, amp = omega_d_amp
        return qt.QobjEvo([self.H0, [amp * self.H1, lambda t: np.cos(omega_d * t)]])
