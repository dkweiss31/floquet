from __future__ import annotations

import functools
import warnings

import numpy as np
import qutip as qt
import scipy as sp

from .model import Model
from .options import Options
from .utils.parallel import parallel_map


class DisplacedState:
    """Class providing methods for computing displaced states.

    Parameters:
        hilbert_dim: Hilbert space dimension
        model: Model including the Hamiltonian, drive amplitudes, frequencies,
            state indices
        state_indices: States of interest
        options: Options used
    """

    def __init__(
        self, hilbert_dim: int, model: Model, state_indices: list, options: Options
    ):
        self.hilbert_dim = hilbert_dim
        self.model = model
        self.state_indices = state_indices
        self.options = options
        self.exponent_pair_idx_map = self._create_exponent_pair_idx_map()

    def overlap_with_bare_states(
        self, amp_idx_0: int, coefficients: np.ndarray, floquet_modes: np.ndarray
    ) -> np.ndarray:
        """Calculate overlap of floquet modes with 'bare' states.

        'Bare' here is defined loosely. For the first range of amplitudes, the bare
        states are truly the bare states (the coefficients are obtained from
        bare_state_coefficients, which give the bare states). For later ranges, we
        define the bare state as the state obtained from the fit from previous range,
        with amplitude evaluated at the lower edge of amplitudes for the new region.
        This is, in a sense, the most natural choice, since it is most analogous to what
        is done in the first window when the overlap is computed against bare
        eigenstates (that obviously don't have amplitude dependence). Moreover, the fit
        coefficients for the previous window by definition were obtained in a window
        that does not include the one we are currently investigating. Asking for the
        state with amplitude values outside of the fit window should be done at your
        own peril.

        Parameters:
            amp_idx_0: Index specifying the lower bound of the amplitude range.
            coefficients: coefficients that specify the bare state that we calculate
                overlaps of Floquet modes against
            floquet_modes: Floquet modes to be compared to the bare states given by
                coefficients
        Returns:
            overlaps with shape (w,a,s) where w is the number of drive frequencies,
                a is the number of drive amplitudes (specified by amp_idxs) and s is the
                number of states we are investigating
        """
        overlaps = np.zeros(floquet_modes.shape[:-1])
        for array_idx, state_idx in enumerate(self.state_indices):
            # Bind the array_idx variable to the function to prevent late-binding
            # closure, see https://docs.python-guide.org/writing/gotchas/#late-binding-closures.
            # This isn't actually a problem in our case but still nice practice to
            # bind the value to the function
            def _compute_bare_state(
                omega_d: float, _array_idx: int = array_idx, _state_idx: int = state_idx
            ) -> np.ndarray:
                omega_d_idx = self.model.omega_d_to_idx(omega_d)
                return self.displaced_state(
                    omega_d,
                    self.model.drive_amplitudes[amp_idx_0, omega_d_idx],
                    _state_idx,
                    coefficients=coefficients[_array_idx],
                ).full()[:, 0]

            bare_states = np.array(
                [_compute_bare_state(omega_d) for omega_d in self.model.omega_d_values],
                dtype=complex,
            )
            # bare states may differ as a function of omega_d, hence the bare states
            # have an index of i that we don't sum over
            # indices are i: omega_d, j: amp, k: components of state
            overlaps[:, :, array_idx] = np.abs(
                np.einsum(
                    "ijk,ik->ij", floquet_modes[:, :, array_idx], np.conj(bare_states)
                )
            )
        return overlaps

    def overlap_with_displaced_states(
        self, amp_idxs: list, coefficients: np.ndarray, floquet_modes: np.ndarray
    ) -> np.ndarray:
        """Calculate overlap of floquet modes with 'ideal' displaced states.

        This is done here for a specific amplitude range.

        Parameters:
            amp_idxs: list of lower and upper amplitude indices specifying the range of
                drive amplitudes this calculation should be done for
            coefficients: coefficients that specify the displaced state that we
                calculate overlaps of Floquet modes against
            floquet_modes: Floquet modes to be compared to the displaced states given by
                coefficients
        Returns:
            overlaps with shape (w,a,s) where w is the number of drive frequencies,
                a is the number of drive amplitudes (specified by amp_idxs) and s is the
                number of states we are investigating
        """

        def _run_overlap_displaced(omega_d_amp: tuple[float, float]) -> np.ndarray:
            overlap = np.zeros(len(self.state_indices))
            omega_d, amp = omega_d_amp
            for array_idx, state_idx in enumerate(self.state_indices):
                floquet_mode_for_idx = floquet_modes[
                    self.model.omega_d_to_idx(omega_d),
                    self.model.amp_to_idx(amp, omega_d),
                    array_idx,
                ]
                disp_state = self.displaced_state(
                    omega_d,
                    amp,
                    state_idx=state_idx,
                    coefficients=coefficients[array_idx],
                ).dag()
                overlap[array_idx] = np.abs(
                    disp_state.data.to_array()[0] @ floquet_mode_for_idx
                )
            return overlap

        omega_d_amp_params = self.model.omega_d_amp_params(amp_idxs)
        amp_range_vals = self.model.drive_amplitudes[amp_idxs[0] : amp_idxs[1]]
        result = list(
            parallel_map(
                self.options.num_cpus, _run_overlap_displaced, omega_d_amp_params
            )
        )
        return np.array(result).reshape(
            (
                len(self.model.omega_d_values),
                len(amp_range_vals),
                len(self.state_indices),
            )
        )

    def bare_state_coefficients(self, state_idx: int) -> np.ndarray:
        r"""For bare state only component is itself.

        Parameters:
            state_idx: Coefficients for the state $|state_idx\rangle$ that when
                evaluated at any amplitude or frequency simply return the bare state.
                Note that this should be the actual state index, and not the array index
                (for instance if we have state_indices=[0, 1, 3] because we're not
                interested in the second excited state, for the 3rd excited state we
                should pass 3 here and not 2).
        """
        coefficient_matrix_for_amp_and_state = np.zeros(
            (self.hilbert_dim, len(self.exponent_pair_idx_map)), dtype=complex
        )
        coefficient_matrix_for_amp_and_state[state_idx, 0] = 1.0
        return coefficient_matrix_for_amp_and_state

    def displaced_state(
        self, omega_d: float, amp: float, state_idx: int, coefficients: np.ndarray
    ) -> qt.Qobj:
        """Construct the ideal displaced state based on a polynomial expansion."""
        return sum(
            self._coefficient_for_state(
                np.array([omega_d, amp]),
                *coefficients[state_idx_component, :],
                bare_same=state_idx == state_idx_component,
            )
            * qt.basis(self.hilbert_dim, state_idx_component)
            for state_idx_component in range(self.hilbert_dim)
        ).unit()

    def _coefficient_for_state(
        self,
        xydata: np.ndarray,
        *state_idx_coefficients: np.ndarray | tuple,
        bare_same: bool = False,
    ) -> np.ndarray | float:
        """Fit function to pass to curve fit, assume a 2D polynomial."""
        exp_pair_map = self.exponent_pair_idx_map
        omega_d, amp = xydata.T
        result = 1.0 if bare_same else 0.0
        result += sum(
            state_idx_coefficients[idx]
            * omega_d ** exp_pair_map[idx][0]
            * amp ** exp_pair_map[idx][1]
            for idx in exp_pair_map
        )
        return result

    def _create_exponent_pair_idx_map(self) -> dict:
        """Create dictionary of terms in polynomial that we fit.

        We truncate the fit if e.g. there is only a single frequency value to scan over
        but the fit is nominally set to order four. We additionally eliminate the
        constant term that should always be either zero or one.
        """
        cutoff_omega_d = min(len(self.model.omega_d_values), self.options.fit_cutoff)
        cutoff_amp = min(len(self.model.drive_amplitudes), self.options.fit_cutoff)
        idx_exp_map = [
            (idx_1, idx_2)
            for idx_1 in range(cutoff_omega_d)
            for idx_2 in range(cutoff_amp)
        ]
        # Kill constant term, which should always be 1 or 0 depending
        # on if the component is the same as the state being fit.
        # Moreover kill any terms depending only on drive frequency, since these
        # coefficients must be zero as the states have to agree at zero drive strength.
        for idx in range(cutoff_omega_d):
            idx_exp_map.remove((idx, 0))
        weighted_vals = [1.01 * idx_1 + idx_2 for (idx_1, idx_2) in idx_exp_map]
        sorted_idxs = np.argsort(weighted_vals)
        sorted_idx_exp_map = {}
        counter = 0
        for sorted_idx in sorted_idxs:
            exponents = idx_exp_map[sorted_idx]
            if sum(exponents) <= self.options.fit_cutoff:
                sorted_idx_exp_map[counter] = exponents
                counter += 1
        return sorted_idx_exp_map


class DisplacedStateFit(DisplacedState):
    """Methods for fitting an ideal displaced state to calculated Floquet modes."""

    def displaced_states_fit(
        self,
        omega_d_amp_slice: list,
        ovlp_with_bare_states: np.ndarray,
        floquet_modes: np.ndarray,
    ) -> np.ndarray:
        """Perform a fit for the indicated range, ignoring specified modes.

        We loop over all states in state_indices and perform the fit for a given
        amplitude range. We ignore floquet modes (not included in the fit) where
        the corresponding value in ovlp_with_bare_states is below the threshold
        specified in options.

        Parameters:
            omega_d_amp_slice: Pairs of omega_d, amplitude values at which the
                floquet modes have been computed and which we will use as the
                independent variables to fit the Floquet modes
            ovlp_with_bare_states: Bare state overlaps that has shape (w, a, s) where w
                is drive frequency, a is drive amplitude and s is state_indices
            floquet_modes: Floquet mode array with the same shape as
                ovlp_with_bare_states except with an additional trailing dimension h,
                the Hilbert-space dimension.

        Returns:
            Optimized fit coefficients
        """

        def _fit_for_state_idx(array_state_idx: tuple[int, int]) -> np.ndarray:
            array_idx, state_idx = array_state_idx
            floquet_mode_for_state = floquet_modes[:, :, array_idx, :]
            mask = ovlp_with_bare_states[:, :, array_idx].ravel()
            # only fit states that we think haven't run into
            # a nonlinear transition (same for omega_d_amp_filtered above)
            omega_d_amp_filtered = [
                omega_d_amp_slice[i]
                for i in range(len(mask))
                if np.abs(mask[i]) > self.options.overlap_cutoff
            ]
            num_coeffs = len(self.exponent_pair_idx_map)
            coefficient_matrix_for_amp_and_state = np.zeros(
                (self.hilbert_dim, num_coeffs), dtype=complex
            )
            if len(omega_d_amp_filtered) < len(self.exponent_pair_idx_map):
                warnings.warn(
                    "Not enough data points to fit. Returning zeros for the fit",
                    stacklevel=3,
                )
                return coefficient_matrix_for_amp_and_state
            for state_idx_component in range(self.hilbert_dim):
                floquet_mode_bare_component = floquet_mode_for_state[
                    :, :, state_idx_component
                ].ravel()
                floquet_component_filtered = floquet_mode_bare_component[
                    np.abs(mask) > self.options.overlap_cutoff
                ]
                bare_same = state_idx_component == state_idx
                bare_component_fit = self._fit_coefficients_for_component(
                    omega_d_amp_filtered, floquet_component_filtered, bare_same
                )
                coefficient_matrix_for_amp_and_state[state_idx_component, :] = (
                    bare_component_fit
                )
            return coefficient_matrix_for_amp_and_state

        array_idxs = np.arange(len(self.state_indices))
        array_state_idxs = zip(array_idxs, self.state_indices, strict=False)
        fit_data = list(
            parallel_map(self.options.num_cpus, _fit_for_state_idx, array_state_idxs)
        )
        return np.array(fit_data, dtype=complex).reshape(
            (
                len(self.state_indices),
                self.hilbert_dim,
                len(self._create_exponent_pair_idx_map()),
            )
        )

    def _fit_coefficients_for_component(
        self,
        omega_d_amp_filtered: list,
        floquet_component_filtered: np.ndarray,
        bare_same: bool,
    ) -> np.ndarray:
        """Fit the floquet modes to an "ideal" displaced state based on a polynomial.

        This is done here over the grid specified by omega_d_amp_slice. We ignore
        floquet mode data indicated by mask, where we suspect by looking at overlaps
        with the bare state that we have hit a resonance.
        """
        p0 = np.zeros(len(self.exponent_pair_idx_map))
        # fit the real and imaginary parts of the overlap separately
        popt_r = self._fit_coefficients_factory(
            omega_d_amp_filtered, np.real(floquet_component_filtered), p0, bare_same
        )
        popt_i = self._fit_coefficients_factory(
            omega_d_amp_filtered,
            np.imag(floquet_component_filtered),
            p0,
            False,  # for the imaginary part, constant term should always be zero
        )
        return popt_r + 1j * popt_i

    def _fit_coefficients_factory(
        self, XYdata: list, Zdata: np.ndarray, p0: tuple | np.ndarray, bare_same: bool
    ) -> np.ndarray:
        poly_fit = functools.partial(self._coefficient_for_state, bare_same=bare_same)
        try:
            popt, _ = sp.optimize.curve_fit(poly_fit, XYdata, Zdata, p0=p0)
        except RuntimeError:
            warnings.warn(
                "fit failed for a bare component, returning zeros for the fit",
                stacklevel=3,
            )
            popt = np.zeros(len(p0))
        return popt
