from __future__ import annotations

import warnings

import dynamiqs as dq
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optimistix as optx
from jax import Array

from .model import Model
from .options import Options
from .utils.utils import create_exponent_pair_idx_map


class DisplacedState:
    """Class providing methods for computing displaced states.

    Parameters:
        hilbert_dim: Hilbert space dimension
        model: Model including the Hamiltonian, drive amplitudes, frequencies,
            state indices
        state_indices: States of interest
        options: Options used
    """

    hilbert_dim: int
    model: Model
    state_indices: list
    options: options
    exponent_pair_idx_map: dict

    def __init__(
        self, hilbert_dim: int, model: Model, state_indices: list, options: Options
    ):
        self.hilbert_dim = hilbert_dim
        self.model = model
        self.state_indices = state_indices
        self.options = options
        cutoff_omega_d = min(len(model.omega_d_values), options.fit_cutoff)
        cutoff_amp = min(len(model.drive_amplitudes), options.fit_cutoff)
        self.exponent_pair_idx_map = create_exponent_pair_idx_map(
            cutoff_omega_d, cutoff_amp, options.fit_cutoff
        )

    # TODO pass in FloquetResult not just modes?
    def overlap_with_bare_states(
        self, amp_idx_0: int, coefficients: Array, floquet_modes: Array
    ) -> Array:
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

        def _compute_overlap(array_state_idx: tuple):
            array_idx, state_idx = array_state_idx
            # Bind the array_idx variable to the function to prevent late-binding
            # closure, see https://docs.python-guide.org/writing/gotchas/#late-binding-closures.
            # This isn't actually a problem in our case but still nice practice to
            # bind the value to the function

            def _compute_bare_state(
                omega_d: float | Array,
                _array_idx: int = array_idx,
                _state_idx: int = state_idx,
            ) -> Array:
                omega_d_idx = self.model.omega_d_to_idx(omega_d)
                return self.displaced_state(
                    omega_d,
                    self.model.drive_amplitudes[amp_idx_0, omega_d_idx],
                    _state_idx,
                    coefficients=coefficients[_array_idx],
                )

            bare_states = jax.vmap(_compute_bare_state)(self.model.omega_d_values)
            # bare states may differ as a function of omega_d, hence the bare states
            # have an index of w that we don't sum over
            # indices are wand for floquet_modes[:, :, array_idx] and wnd for bare_states
            # where: w; omega_d, a; amp, n; components of state, d; dimension 1
            return dq.dag(floquet_modes[:, :, array_idx]) @ bare_states[:, None]

        return jax.vmap(_compute_overlap)(enumerate(self.state_indices))

    def overlap_with_displaced_states(
        self, amp_idxs: list, coefficients: Array, floquet_modes: Array
    ) -> Array:
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

        def _run_overlap_displaced(omega_d: float | Array, amp: float | Array) -> Array:
            def _compute_overlap(array_state_idx: tuple):
                array_idx, state_idx = array_state_idx
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
                )
                return jnp.abs(dq.dag(floquet_mode_for_idx) @ disp_state)

            return jax.vmap(_compute_overlap)(enumerate(self.state_indices))

        amp_range_vals = self.model.drive_amplitudes[amp_idxs[0] : amp_idxs[1]]
        return jax.vmap(_run_overlap_displaced, in_axes=(0, 1))(
            self.model.omega_d_values, amp_range_vals
        )

    def bare_state_coefficients(self, state_idx: int) -> Array:
        r"""For bare state only component is itself.

        Parameters:
            state_idx: Coefficients for the state $|state_idx\rangle$ that when
                evaluated at any amplitude or frequency simply return the bare state.
                Note that this should be the actual state index, and not the array index
                (for instance if we have state_indices=[0, 1, 3] because we're not
                interested in the second excited state, for the 3rd excited state we
                should pass 3 here and not 2).
        """
        coefficient_matrix = jnp.zeros(
            (self.hilbert_dim, len(self.exponent_pair_idx_map)), dtype=complex
        )
        return coefficient_matrix.at[state_idx, 0].set(1.0)

    def displaced_state(
        self, omega_d: float, amp: float, state_idx: int, coefficients: Array
    ) -> Array:
        """Construct the ideal displaced state based on a polynomial expansion."""

        def _state_component(carry, state_idx_component):
            coefficient = self._coefficient_for_state(
                *coefficients[state_idx_component, :],
                np.array([omega_d, amp]),
                bare_same=state_idx == state_idx_component,
            )
            component = coefficient * dq.basis(self.hilbert_dim, state_idx_component)
            return carry + component, component

        init_state = 0.0 * dq.basis(self.hilbert_dim, 0)
        total_state, _ = jax.lax.scan(
            _state_component, init_state, jnp.arange(self.hilbert_dim)
        )
        return dq.unit(total_state)

    def _coefficient_for_state(
        self,
        *state_idx_coefficients: Array | tuple,
        xydata: Array,
        bare_same: bool = False,
    ) -> Array | float:
        """Fit function to pass to curve fit, assume a 2D polynomial."""
        exp_pair_map = self.exponent_pair_idx_map
        # could be a pair of floats or a pair of vectors
        omega_d, amp = xydata.T

        def _evaluate(carry, idx):
            idx_contribution = (
                state_idx_coefficients[idx]
                * omega_d ** exp_pair_map[idx][0]
                * amp ** exp_pair_map[idx][1]
            )
            carry = carry + idx_contribution
            return carry, idx_contribution

        result, _ = jax.lax.scan(_evaluate, 0.0 * omega_d, xs=exp_pair_map)
        return jnp.where(bare_same, 1.0 + result, result)


class DisplacedStateFit(DisplacedState):
    """Methods for fitting an ideal displaced state to calculated Floquet modes."""

    def displaced_states_fit(
        self,
        omega_d_amp_slice: list,
        ovlp_with_bare_states: Array,
        floquet_modes: Array,
    ) -> Array:
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

        def _fit_for_state_idx(array_state_idx: enumerate) -> Array:
            array_idx, state_idx = array_state_idx
            floquet_mode_for_state = floquet_modes[:, :, array_idx, :]
            mask = (
                jnp.abs(ovlp_with_bare_states[:, :, array_idx].ravel())
                > self.options.fit_cutoff
            )
            # only fit states that we think haven't run into a nonlinear transition
            omega_d_amp_masked = jnp.where(mask, omega_d_amp_slice, jnp.nan)
            num_coeffs = len(self.exponent_pair_idx_map)
            num_not_nan = jnp.sum(not jnp.isnan(omega_d_amp_masked))
            if num_not_nan < len(self.exponent_pair_idx_map):
                # TODO not jax compatible
                warnings.warn(
                    "Not enough data points to fit. Returning zeros for the fit",
                    stacklevel=3,
                )
                return jnp.zeros((self.hilbert_dim, num_coeffs), dtype=complex)

            def _fit_for_component(state_idx_component):
                floquet_mode_bare_component = floquet_mode_for_state[
                    :, :, state_idx_component
                ].ravel()
                floquet_mode_bare_component_masked = jnp.where(
                    mask, floquet_mode_bare_component, jnp.nan
                )
                bare_same = state_idx_component == state_idx
                return self._fit_coefficients_for_component(
                    omega_d_amp_masked, floquet_mode_bare_component_masked, bare_same
                )

            return jax.vmap(_fit_for_component)(range(self.hilbert_dim))

        return jax.vmap(_fit_for_state_idx)(enumerate(self.state_indices))

    def _fit_coefficients_for_component(
        self,
        omega_d_amp_masked: Array,
        floquet_component_masked: Array,
        bare_same: bool,
    ) -> Array:
        """Fit the floquet modes to an "ideal" displaced state based on a polynomial.

        This is done here over the grid specified by omega_d_amp_slice. We ignore
        floquet mode data indicated by mask, where we suspect by looking at overlaps
        with the bare state that we have hit a resonance.
        """
        p0 = jnp.zeros(len(self.exponent_pair_idx_map))
        # fit the real and imaginary parts of the overlap separately
        popt_r = self._fit_coefficients_factory(
            omega_d_amp_masked, jnp.real(floquet_component_masked), p0, bare_same
        )
        popt_i = self._fit_coefficients_factory(
            omega_d_amp_masked,
            jnp.imag(floquet_component_masked),
            p0,
            False,  # for the imaginary part, constant term should always be zero
        )
        return popt_r + 1j * popt_i

    def _fit_coefficients_factory(
        self, XY_data: list, Z_data: Array, p0: tuple | Array, bare_same: bool
    ) -> Array:
        # TODO solver to use here? LM?

        def _residuals(_p0):
            coefficient_fun = jtu.Partial(
                self._coefficient_for_state, *_p0, bare_same=bare_same
            )
            pred_Z_data = jax.vmap(coefficient_fun)(XY_data)
            return jnp.nansum(jnp.abs(Z_data - pred_Z_data) ** 2)

        try:
            # solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
            solver = optx.BFGS(rtol=1e-8, atol=1e-8)
            opt_result = optx.minimise(_residuals, solver, p0)
        except RuntimeError:
            # TODO not jax compatible, think about exception handling
            warnings.warn(
                "fit failed for a bare component, returning zeros for the fit",
                stacklevel=3,
            )
            opt_result = jnp.zeros(len(p0))
        return opt_result
