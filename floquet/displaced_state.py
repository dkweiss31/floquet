from __future__ import annotations

import warnings

import numpy as np

from .model import Model
from .options import Options


class DisplacedState:
    """Class providing methods for computing displaced states.

    Parameters:
        hilbert_dim: Hilbert space dimension
        model: Model; stores the Hamiltonian, drive amplitudes, and frequencies
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
        self.exponent_pairs = self._create_exponent_pairs()
        self.poly_terms = self._create_poly_terms()

    def overlap_with_bare_states(
        self,
        coefficients: np.ndarray,
        floquet_modes: np.ndarray,
        omega_d_idxs: tuple[int, int] = (0, None),
        amp_idx_0: int = 0,
    ) -> np.ndarray:
        """Calculate overlap of floquet modes with 'bare' states.

        'Bare' here is defined loosely. For the first range of amplitudes, the bare
        states are truly the bare states (all zero coefficients). For later ranges, we
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
            coefficients: coefficients that specify the bare state that we calculate
                overlaps of Floquet modes against.
                Shape: `(num_states, hilbert_dim, num_fit_terms)`.
            floquet_modes: Floquet modes to be compared to the bare states given by
                coefficients. The -2 dimension here indexes the modes and the
                -1 dimension indexes the hilbert space.
                Shape: `(num_omega_d, num_amps, hilbert_dim, hilbert_dim)`.
            omega_d_idxs: Indices specifying the lower and upper bound of the drive
                frequency range. Selects all `self.model.omega_d_values` by default.
            amp_idx_0: Index specifying the lower bound of the amplitude range.
                0 by default, i.e. selects the undriven states.

        Returns:
            Overlaps between the floquet modes and the bare states for the
                corresponding drive frequencies and the amplitude value at the lower
                edge of the amplitude range, specified by amp_idx_0.
                Shape: `(num_omega_d, num_amps, num_states)`.
        """
        floquet_modes = floquet_modes[omega_d_idxs[0] : omega_d_idxs[1], ...]
        displaced_states = self.displaced_state(
            coefficients, omega_d_idxs, [amp_idx_0, amp_idx_0 + 1]
        )[:, 0, ...]

        return np.abs(
            np.einsum("wsh,wash->was", np.conj(displaced_states), floquet_modes)
        )

    def overlap_with_displaced_states(
        self,
        coefficients: np.ndarray,
        floquet_modes: np.ndarray,
        omega_d_idxs: tuple[int, int] = (0, None),
        amp_idxs: tuple[int, int] = (0, None),
    ) -> np.ndarray:
        """Calculate overlap of floquet modes with 'ideal' displaced states.

        This is done here for a specific amplitude range.

        Parameters:
            coefficients: coefficients that specify the displaced state that we
                calculate overlaps of Floquet modes against.
                Shape: `(num_states, hilbert_dim, num_fit_terms)`.
            floquet_modes: Floquet modes to be compared to the bare states given by
                coefficients.
                Shape: `(num_omega_ds, num_amps, num_states, hilbert_dim)`.
            omega_d_idxs: Indices specifying the lower and upper bound of the drive
                frequency range. Selects all `self.model.omega_d_values` by default.
            amp_idxs: Indices specifying the lower and upper bound of the amplitude
                range. Selects all self.model.drive_amplitudes by default.

        Returns:
            Overlaps between the floquet mode and the corresponding displaced state,
                for each drive frequency and drive amplitude.
                Shape: `(num_omega_ds, num_amps, num_states)`.
        """
        floquet_modes = floquet_modes[
            omega_d_idxs[0] : omega_d_idxs[1], amp_idxs[0] : amp_idxs[1], :, :
        ]
        displaced_states = self.displaced_state(coefficients, omega_d_idxs, amp_idxs)
        return np.abs(
            np.einsum("wash,wash->was", np.conj(displaced_states), floquet_modes)
        )

    def displaced_state(
        self,
        coefficients: np.ndarray,
        omega_d_idxs: tuple[int, int] = (0, None),
        amp_idxs: tuple[int, int] = (0, None),
    ) -> np.ndarray:
        r"""Construct the ideal displaced state.

        The displaced state is given by a low-order expansion around the corresponding
        bare state. For a drive frequency $\omega_d$ and drive amplitude $\Omega_d$,
        the displaced state is given by,
        $$
        \left| \tilde{n} (\omega_d, \Omega_d) \right> = \left| n \right>
        + \sum_l \sum_{k_0, k_1} c_{n, l, k_0, k_1}
        \omega_d^{k_0} \Omega_d^{k_1} \left| l \right>
        $$
        where the coefficients $c_{n, l, k_0, k_1}$ are provided in the `coefficients`
        argument. The constant term (i.e. $k_0 = k_1 = 0$) is excluded from the fit.
        Note, that the $(k_0, k_1)$ pairs are stacked, and provided by
        `self.exponent_pairs`.

        Parameters:
            coefficients: Coefficients to expand the displaced state in terms of the
                undriven states. Shape: `(num_states, hilbert_dim, num_fit_terms)`.
            omega_d_idxs: Indices specifying the lower and upper bound of the drive
                frequency range. Selects all `self.model.omega_d_values` by default.
            amp_idxs: Indices specifying the lower and upper bound of the amplitude
                range. Selects all self.model.drive_amplitudes by default. Selects
                all self.model.drive_amplitudes by default.

        Returns:
            The displaced state(s) for the specified range of drive frequencies and
                drive amplitudes. Shape: `(num_omega_ds, num_amps, num_states,
                hilbert_dim)`.
                Careful, that the num_states index is the array index, not the state
                index! For example, if state_indices = [0, 2], then result[0, ...] is
                the displaced state for state index 0, and result[1, ...] is the
                displaced state for state index 2.
        """
        # Tensor of polynomial terms omega^{k_0} * amp^{k_1}.
        # Shape: (num_omega_ds, num_amps, num_fit_terms)
        _poly_terms = self.poly_terms[
            omega_d_idxs[0] : omega_d_idxs[1], amp_idxs[0] : amp_idxs[1], :
        ]

        # Compute the perturbation, based on the given coefficients.
        result = np.einsum("wat,sht->wash", _poly_terms, coefficients)

        # Add the perturbation to the bare state. Bare states are defined by the model.
        # Get only the states corresponding to state_indices, and tile them for all
        # omega_ds and amps.
        result += np.tile(
            self.model.bare_state_array()[None, None, self.state_indices, :],
            (*result.shape[:2], 1, 1),
        )

        # Normalize
        result /= np.linalg.norm(result, axis=-1, keepdims=True)

        return result

    def _create_poly_terms(self) -> np.ndarray:
        r"""Compute a tensor of polynomial terms $\omega_d^{k_0} * \Omega_d^{k_1}$."""
        omega_power = (
            self.model.omega_d_values[:, None, None]
            ** self.exponent_pairs[0][None, None, :]
        )
        amp_power = (
            self.model.drive_amplitudes.T[:, :, None]
            ** self.exponent_pairs[1][None, None, :]
        )
        return omega_power * amp_power

    def _create_exponent_pairs(self) -> np.ndarray:
        """Create dictionary of terms in polynomial that we fit.

        We truncate the fit if e.g. there is only a single frequency value to scan over
        but the fit is nominally set to order four. We additionally eliminate the
        constant term that should always be either zero or one.
        """
        cutoff_omega_d = min(len(self.model.omega_d_values), self.options.fit_cutoff)
        cutoff_amp = min(len(self.model.drive_amplitudes), self.options.fit_cutoff)
        # Generate all combinations of indices.
        # Remove amplitude-independent terms (i.e. when the exponent for the amp == 0).
        # This is enforced by the fact the states have to agree at zero drive strength.
        idx_exp_map = np.stack(
            np.meshgrid(
                np.arange(cutoff_omega_d), np.arange(1, cutoff_amp), indexing="ij"
            ),
            axis=-1,
        ).reshape(-1, 2)

        # Only keep terms where the sum of the exponents is less than the cutoff
        idx_exp_map = idx_exp_map[
            np.sum(idx_exp_map, axis=-1) <= self.options.fit_cutoff
        ]

        # Sort. Introduce a fudge factor to ensure that the sorting is stable
        # Only keep terms where the sum of the exponents is less than the cutoff
        weighted_vals = 1.01 * idx_exp_map[:, 0] + idx_exp_map[:, 1]
        sorted_idxs = np.argsort(weighted_vals)
        return idx_exp_map[sorted_idxs].T


class DisplacedStateFit(DisplacedState):
    """Methods for fitting an ideal displaced state to calculated Floquet modes."""

    def displaced_states_fit(
        self,
        ovlp_with_bare_states: np.ndarray,
        floquet_modes: np.ndarray,
        omega_d_idxs: tuple[int, int] = (0, None),
        amp_idxs: tuple[int, int] = (0, None),
    ) -> np.ndarray:
        """Perform a fit for the indicated range, ignoring specified modes.

        We loop over all states in state_indices and perform the fit for a given
        amplitude range. We ignore floquet modes (not included in the fit) where
        the corresponding value in ovlp_with_bare_states is below the threshold
        specified in options.

        Parameters:
            ovlp_with_bare_states: Overlap with bare states. Already trimmed to the
                relevant range of frequencies and amplitudes.
                Shape: `(num_omega_d_in_range, num_amps_in_range, num_states)`.
            floquet_modes: Floquet modes.
                Shape: `(num_omega_ds, num_amps, num_states, hilbert_dim)`.
            omega_d_idxs: Indices specifying the lower and upper bound of the drive
                frequency range. Selects all `self.model.omega_d_values` by default.
            amp_idxs: Indices specifying the lower and upper bound of the amplitude
                range. Selects all self.model.drive_amplitudes by default.

        Returns:
            Optimized fit coefficients.
                Shape: `(num_states, hilbert_dim, num_fit_terms)`.
        """
        # Only fit states that we think haven't run into a transition,
        # and those within the specified bounds
        # mask.shape = (num_omega_ds, num_amps, num_state_indices).
        mask = ovlp_with_bare_states > self.options.overlap_cutoff

        # Required poly_terms
        poly_terms = self.poly_terms[
            omega_d_idxs[0] : omega_d_idxs[1], amp_idxs[0] : amp_idxs[1], :
        ]

        coeffs = np.zeros(
            (len(self.state_indices), self.hilbert_dim, self.exponent_pairs.shape[-1]),
            dtype=complex,
        )

        # Fit each state separately, since their masks may differ
        for arr_idx, state_idx in enumerate(self.state_indices):
            coeffs[arr_idx] = self._fit_for_state_idx(
                target_states=floquet_modes[..., arr_idx, :],
                mask=mask[..., arr_idx],
                state_index=state_idx,
                poly_terms=poly_terms,
            )

        # TODO: return mask, other fit data
        return coeffs

    def _fit_for_state_idx(
        self,
        target_states: np.ndarray,
        mask: np.ndarray,
        state_index: int,
        poly_terms: np.ndarray,
    ) -> np.ndarray:
        num_fit_terms = self.exponent_pairs.shape[-1]
        mask_flat = mask.ravel()

        # Warn if not enough data points to fit
        if np.count_nonzero(mask_flat) < num_fit_terms:
            warnings.warn(
                "Not enough data points to fit. Returning zeros for the fit",
                stacklevel=3,
            )
            return np.zeros((self.hilbert_dim, num_fit_terms), dtype=complex)

        # Flatten and mask arrays. Resulting rows index masked amp-freq pairs.
        # For the poly_terms matrix, the cols index the fit terms.
        # For states, the cols index the hilbert_dim
        masked_poly_terms = poly_terms.reshape(-1, num_fit_terms)[mask_flat]
        masked_target_states = target_states.reshape(-1, self.hilbert_dim)[mask_flat]

        # Bare states array (repeated for all amp-freq pairs)
        masked_bare_states = np.tile(
            self.model.bare_state_array()[None, state_index, :],
            (masked_target_states.shape[0], 1),
        )

        # Fit the difference between the target states and the bare states
        masked_states_to_fit = masked_target_states - masked_bare_states

        if np.any(np.abs(masked_states_to_fit) > 1):
            warnings.warn(
                "Values to large. Fit may be unreliable. Returning zeros for the fit.",
                stacklevel=3,
            )
            return np.zeros((self.hilbert_dim, num_fit_terms), dtype=complex)

        # Simple linear fit: masked_states_to_fit = masked_poly_terms @ coefficients.T
        try:
            popt = np.linalg.lstsq(masked_poly_terms, masked_states_to_fit)[0].T

        except np.linalg.LinAlgError:
            warnings.warn("Fit failed. Returning zeros for the fit", stacklevel=3)
            popt = np.zeros((self.hilbert_dim, num_fit_terms), dtype=complex)

        return popt
