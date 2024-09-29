from __future__ import annotations

import ast
import functools
import itertools
import time
import warnings
from itertools import chain, product

import numpy as np
import qutip as qt
import scipy as sp

from .options import Options
from .utils.file_io import extract_info_from_h5, update_data_in_h5, write_to_h5
from .utils.parallel import parallel_map


def floquet_analysis(
    H0: qt.Qobj | np.ndarray | list,
    H1: qt.Qobj | np.ndarray | list,
    drive_parameters: DriveParameters,
    state_indices: list | None = None,
    options: Options = Options(),  # noqa B008
    init_data_to_save: dict | None = None,
) -> FloquetAnalysis:
    """Perform a floquet analysis to identify nonlinear resonances.

    Arguments:
        H0: Drift Hamiltonian (ideally diagonal)
        H1: Drive operator
        drive_parameters: Class specifying the drive amplitudes and frequencies
        state_indices: State indices of interest
        options: Options for the Floquet analysis.
        init_data_to_save: Initial parameter metadata to save to file. Defaults to None.

    Returns:
        FloquetAnalysis object on which we can call run() to perform the full floquet
            simulation.
    """
    if state_indices is None:
        state_indices = [0, 1]
    if not isinstance(H0, qt.Qobj):
        H0 = qt.Qobj(np.array(H0, dtype=complex))
    if not isinstance(H1, qt.Qobj):
        H1 = qt.Qobj(np.array(H1, dtype=complex))
    return FloquetAnalysis(
        H0,
        H1,
        drive_parameters,
        state_indices,
        options,
        init_data_to_save=init_data_to_save,
    )


def floquet_analysis_from_file(filepath: str) -> FloquetAnalysis:
    """Reinitialize a FloquetAnalysis object from file.

    Here we only reinitialize the input parameters and not the computed data.

    Arguments:
        filepath: Path to the file

    Returns:
        FloquetAnalysis object with identical initial parameters to the one previously
            written to file.
    """
    _, param_dict = extract_info_from_h5(filepath)
    floquet_init = ast.literal_eval(param_dict["floquet_analysis_init"])
    return floquet_analysis(
        floquet_init["H0"],
        floquet_init["H1"],
        drive_parameters=DriveParameters(**floquet_init["drive_parameters"]),
        state_indices=floquet_init["state_indices"],
        options=Options(**floquet_init["options"]),
        init_data_to_save=floquet_init["init_data_to_save"],
    )


class DriveParameters:
    """Class that handles the drive strength and frequency.

    Parameters:
        omega_d_values: drive frequencies to scan over
        drive_amplitudes: amp values to scan over. Can be one dimensional in which case
            these amplitudes are used for all omega_d, or it can be two dimensional
            in which case the first dimension are the amplitudes to scan over
            and the second are the amplitudes for respective drive frequencies
    """

    def __init__(self, omega_d_values: np.ndarray, drive_amplitudes: np.ndarray):
        if isinstance(omega_d_values, list):
            omega_d_values = np.array(omega_d_values)
        if isinstance(drive_amplitudes, list):
            drive_amplitudes = np.array(drive_amplitudes)
        if len(drive_amplitudes.shape) == 1:
            drive_amplitudes = np.tile(drive_amplitudes, (len(omega_d_values), 1)).T
        else:
            assert len(drive_amplitudes.shape) == 2
            assert drive_amplitudes.shape[1] == len(omega_d_values)
        self.omega_d_values = omega_d_values
        self.drive_amplitudes = drive_amplitudes

    def omega_d_to_idx(self, omega_d: float) -> np.ndarray[int]:
        """Return index corresponding to omega_d value."""
        return np.argmin(np.abs(self.omega_d_values - omega_d))

    def amp_to_idx(self, amp: float, omega_d: float) -> np.ndarray[int]:
        """Return index corresponding to amplitude value."""
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


class DisplacedState:
    """Class providing methods for computing displaced states.

    Parameters:
        hilbert_dim: Hilbert space dimension
        drive_parameters: Drive parameters used
        state_indices: States of interest
        options: Options used
    """

    def __init__(
        self,
        hilbert_dim: int,
        drive_parameters: DriveParameters,
        state_indices: list,
        options: Options,
    ):
        self.hilbert_dim = hilbert_dim
        self.drive_parameters = drive_parameters
        self.state_indices = state_indices
        self.options = options
        self.exponent_pair_idx_map = self._create_exponent_pair_idx_map()

    def overlap_with_bare_states(
        self, amp_idx_0: int, coefficients: np.ndarray, floquet_data: np.ndarray
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
            floquet_data: Floquet data to be compared to the bare states given by
                coefficients
        Returns:
            overlaps with shape (w,a,s) where w is the number of drive frequencies,
                a is the number of drive amplitudes (specified by amp_idxs) and s is the
                number of states we are investigating
        """
        overlaps = np.zeros(floquet_data.shape[:-1])
        for array_idx, state_idx in enumerate(self.state_indices):
            # Bind the array_idx variable to the function to prevent late-binding
            # closure, see https://docs.python-guide.org/writing/gotchas/#late-binding-closures.
            # This isn't actually a problem in our case but still nice practice to
            # bind the value to the function
            def _compute_bare_state(
                omega_d: float, _array_idx: int = array_idx, _state_idx: int = state_idx
            ) -> np.ndarray:
                omega_d_idx = self.drive_parameters.omega_d_to_idx(omega_d)
                return self.displaced_state(
                    omega_d,
                    self.drive_parameters.drive_amplitudes[amp_idx_0, omega_d_idx],
                    _state_idx,
                    coefficients=coefficients[_array_idx],
                ).full()[:, 0]

            bare_states = np.array(
                [
                    _compute_bare_state(omega_d)
                    for omega_d in self.drive_parameters.omega_d_values
                ],
                dtype=complex,
            )
            # bare states may differ as a function of omega_d, hence the bare states
            # have an index of i that we don't sum over
            # indices are i: omega_d, j: amp, k: components of state
            overlaps[:, :, array_idx] = np.abs(
                np.einsum(
                    "ijk,ik->ij", floquet_data[:, :, array_idx], np.conj(bare_states)
                )
            )
        return overlaps

    def overlap_with_displaced_states(
        self, amp_idxs: list, coefficients: np.ndarray, floquet_data: np.ndarray
    ) -> np.ndarray:
        """Calculate overlap of floquet modes with 'ideal' displaced states.

        This is done here for a specific amplitude range.

        Parameters:
            amp_idxs: list of lower and upper amplitude indices specifying the range of
                drive amplitudes this calculation should be done for
            coefficients: coefficients that specify the displaced state that we
                calculate overlaps of Floquet modes against
            floquet_data: Floquet data to be compared to the displaced states given by
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
                floquet_data_for_idx = floquet_data[
                    self.drive_parameters.omega_d_to_idx(omega_d),
                    self.drive_parameters.amp_to_idx(amp, omega_d),
                    array_idx,
                ]
                disp_state = self.displaced_state(
                    omega_d,
                    amp,
                    state_idx=state_idx,
                    coefficients=coefficients[array_idx],
                ).dag()
                overlap[array_idx] = np.abs(
                    disp_state.data.toarray()[0] @ floquet_data_for_idx
                )
            return overlap

        omega_d_amp_params = self.drive_parameters.omega_d_amp_params(amp_idxs)
        amp_range_vals = self.drive_parameters.drive_amplitudes[
            amp_idxs[0] : amp_idxs[1]
        ]
        result = list(
            parallel_map(
                self.options.num_cpus, _run_overlap_displaced, omega_d_amp_params
            )
        )
        return np.array(result).reshape(
            (
                len(self.drive_parameters.omega_d_values),
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
        cutoff_omega_d = min(
            len(self.drive_parameters.omega_d_values), self.options.fit_cutoff
        )
        cutoff_amp = min(
            len(self.drive_parameters.drive_amplitudes), self.options.fit_cutoff
        )
        idx_exp_map = [
            (idx_1, idx_2)
            for idx_1 in range(cutoff_omega_d)
            for idx_2 in range(cutoff_amp)
        ]
        # kill constant term, which should always be 1 or 0 depending
        # on if the component is the same as the state being fit
        idx_exp_map.remove((0, 0))
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
        omega_d_amp_data_slice: list,
        ovlp_with_bare_states: np.ndarray,
        floquet_data: np.ndarray,
    ) -> np.ndarray:
        """Perform a fit for the indicated range, ignoring specified modes.

        We loop over all states in state_indices and perform the fit for a given
        amplitude range. We ignore floquet modes (not included in the fit) where
        the corresponding value in ovlp_with_bare_states is below the threshold
        specified in options.

        Parameters:
            omega_d_amp_data_slice: Pairs of omega_d, amplitude values at which the
                floquet modes have been computed and which we will use as the
                independent variables to fit the Floquet modes
            ovlp_with_bare_states: Bare state overlaps that has shape (w, a, s) where w
                is drive frequency, a is drive amplitude and s is state_indices
            floquet_data: Floquet mode array with the same shape as
                ovlp_with_bare_states except with an additional trailing dimension h,
                the Hilbert-space dimension.

        Returns:
            Optimized fit coefficients
        """

        def _fit_for_state_idx(array_state_idx: tuple[int, int]) -> np.ndarray:
            array_idx, state_idx = array_state_idx
            floquet_idx_data = floquet_data[:, :, array_idx, :]
            mask = ovlp_with_bare_states[:, :, array_idx].ravel()
            # only fit states that we think haven't run into
            # a nonlinear transition (same for omega_d_amp_filtered above)
            omega_d_amp_filtered = [
                omega_d_amp_data_slice[i]
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
                floquet_idx_data_bare_component = floquet_idx_data[
                    :, :, state_idx_component
                ].ravel()
                floquet_component_filtered = floquet_idx_data_bare_component[
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

        This is done here over the grid specified by omega_d_amp_data_slice. We ignore
        floquet data indicated by mask, where we suspect by looking at overlaps with the
        bare state that we have hit a resonance.
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


class FloquetAnalysis:
    def __init__(
        self,
        H0: qt.Qobj,
        H1: qt.Qobj,
        drive_parameters: DriveParameters,
        state_indices: list,
        options: Options,
        init_data_to_save: dict | None = None,
    ):
        self.H0 = H0
        self.H1 = H1
        self.drive_parameters = drive_parameters
        self.state_indices = state_indices
        self.options = options
        self.init_data_to_save = init_data_to_save
        # Save in _init_attrs for later re-initialization. Everything added to self
        # after this is a derived quantity
        self._init_attrs = set(self.__dict__.keys())
        self.hilbert_dim = H0.shape[0]

    def get_initdata(self) -> dict:
        """Collect all init attributes for writing to file."""
        init_dict = {k: v for k, v in self.__dict__.items() if k in self._init_attrs}
        new_init_dict = {}
        for k, v in init_dict.items():
            if isinstance(v, qt.Qobj):
                new_init_dict[k] = v.data.toarray().tolist()
            elif isinstance(v, np.ndarray):
                new_init_dict[k] = v.tolist()
            elif isinstance(v, Options):
                new_init_dict[k] = vars(v)
            elif isinstance(v, DriveParameters):
                dp_dict = {}
                for dp_key, dp_val in vars(v).items():
                    dp_dict[dp_key] = dp_val.tolist()
                new_init_dict[k] = dp_dict
            else:
                new_init_dict[k] = v
        return new_init_dict

    def param_dict(self) -> dict:
        """Collect all attributes for writing to file, including derived ones."""
        if self.init_data_to_save is None:
            self.init_data_to_save = {}
        return (
            vars(self.options)
            | vars(self.drive_parameters)
            | self.init_data_to_save
            | {
                "hilbert_dim": self.hilbert_dim,
                "floquet_analysis_init": self.get_initdata(),
            }
        )

    def __str__(self) -> str:
        params = self.param_dict()
        params.pop("floquet_analysis_init")
        parts_str = "\n".join(f"{k}: {v}" for k, v in params.items() if v is not None)
        return "Running floquet simulation with parameters: \n" + parts_str

    def hamiltonian(self, params: tuple[float, float]) -> list[qt.Qobj]:
        """Return the Hamiltonian we actually simulate."""
        omega_d, amp = params
        return [self.H0, [amp * self.H1, lambda t, _: np.cos(omega_d * t)]]

    def run_one_floquet(
        self, omega_d_amp: tuple[float, float]
    ) -> tuple[np.ndarray, qt.Qobj]:
        """Run one instance of the problem for a pair of drive frequency and amp.

        Returns floquet modes as numpy column vectors, as well as the quasienergies.

        Parameters:
            omega_d_amp: Pair of drive frequency and amp.
        """
        omega_d, _ = omega_d_amp
        T = 2.0 * np.pi / omega_d
        f_modes_0, f_energies_0 = qt.floquet_modes(
            self.hamiltonian(omega_d_amp),  # type: ignore
            T,
            options=qt.Options(nsteps=self.options.nsteps),
        )
        sampling_time = self.options.floquet_sampling_time_fraction * T % T
        if sampling_time != 0:
            f_modes_t = qt.floquet_modes_t(
                f_modes_0,
                f_energies_0,
                sampling_time,
                self.hamiltonian(omega_d_amp),  # type: ignore
                T,
                options=qt.Options(nsteps=self.options.nsteps),
            )
        else:
            f_modes_t = f_modes_0
        return f_modes_t, f_energies_0

    def calculate_modes_quasies_ovlps(
        self,
        f_modes_energies: tuple[np.ndarray, qt.Qobj],
        params_0: tuple[float, float],
        displaced_state: DisplacedState,
        previous_coefficients: np.ndarray,
    ) -> np.ndarray:
        """Return overlaps with "ideal" bare state at a given pair of (omega_d, amp).

        Parameters:
            f_modes_energies: output of self.run_one_floquet(params)
            params_0: (omega_d_0, amp_0) to use for displaced fit
            displaced_state: Instance of DisplacedState
            previous_coefficients: Coefficients from the previous amplitude range that
                will be used when calculating overlap of the floquet modes against
                the 'bare' states specified by previous_coefficients

        """
        f_modes_0, _ = f_modes_energies
        # construct column vectors to compute overlaps
        f_modes_cols = np.array(
            [f_modes_0[idx].data.toarray()[:, 0] for idx in range(self.hilbert_dim)],
            dtype=complex,
        ).T
        # return overlap and floquet mode
        modes_quasies_ovlps = np.zeros(
            (len(self.state_indices), 1 + self.hilbert_dim), dtype=complex
        )
        # TODO: refactor using overlap_with_bare_states method?
        ideal_displaced_state_array = np.array(
            [
                displaced_state.displaced_state(
                    *params_0, state_idx, previous_coefficients[array_idx]
                )
                .dag()
                .data.toarray()[0]
                for array_idx, state_idx in enumerate(self.state_indices)
            ]
        )
        overlaps = np.einsum("ij,jk->ik", ideal_displaced_state_array, f_modes_cols)
        # take the argmax along k
        f_idxs = np.argmax(np.abs(overlaps), axis=1)
        for array_idx, _state_idx in enumerate(self.state_indices):
            f_idx = f_idxs[array_idx]
            bare_state_overlap = overlaps[array_idx, f_idx]
            modes_quasies_ovlps[array_idx, 0] = bare_state_overlap
            modes_quasies_ovlps[array_idx, 1:] = (
                np.sign(bare_state_overlap) * f_modes_cols[:, f_idx]
            )
        return modes_quasies_ovlps

    def bare_state_array(self) -> np.ndarray:
        """Return array of bare states.

        Used to specify initial bare states for the Blais branch analysis.
        """
        return np.squeeze(
            np.array(
                [qt.basis(self.hilbert_dim, idx) for idx in range(self.hilbert_dim)]
            )
        )

    def _step_in_amp(
        self, f_modes_energies: tuple[np.ndarray, qt.Qobj], prev_f_modes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform Blais branch analysis.

        Gorgeous in its simplicity. Simply calculate overlaps of new floquet modes with
        those from the previous amplitude step, and order the modes accordingly. So
        ordered, compute the mean excitation number, yielding our branches.
        """
        f_modes_0, f_energies_0 = f_modes_energies
        f_modes_0 = np.squeeze(np.array(f_modes_0))
        all_overlaps = np.abs(np.einsum("ij,kj->ik", np.conj(prev_f_modes), f_modes_0))
        # assume that prev_f_modes_arr have been previously sorted. Question
        # is which k index has max overlap?
        max_idxs = np.argmax(all_overlaps, axis=1)
        f_modes_ordered = f_modes_0[max_idxs]
        avg_excitation = self._calculate_mean_excitation(f_modes_ordered)
        sorted_quasi_es = f_energies_0[max_idxs]
        return avg_excitation, sorted_quasi_es, f_modes_ordered

    def _calculate_mean_excitation(self, f_modes_ordered: np.ndarray) -> np.ndarray:
        """Mean excitation number of ordered floquet modes.

        Based on Blais arXiv:2402.06615, specifically Eq. (12) but going without the
        integral over floquet modes in one period.
        """
        bare_states = self.bare_state_array()
        overlaps_sq = np.abs(np.einsum("ij,kj->ik", bare_states, f_modes_ordered)) ** 2
        # sum over bare excitations weighted by excitation number
        return np.einsum("ik,i->k", overlaps_sq, np.arange(0, self.hilbert_dim))

    def run(self, filepath: str = "tmp.h5py") -> dict:
        """Perform floquet analysis over range of amplitudes and drive frequencies.

        This function largely performs two calculations. The first is the Xiao analysis
        introduced in https://arxiv.org/abs/2304.13656, fitting the extracted Floquet
        modes to the "ideal" displaced state which does not include resonances by design
        (because we fit to a low order polynomial and ignore any floquet modes with
        overlap with the bare state below a given threshold). This analysis produces the
        "scar" plots. The second is the Blais branch analysis, which tracks the Floquet
        modes by stepping in drive amplitude for a given drive frequency. For this
        reason the code is structured to parallelize over drive frequency, but scans in
        a loop over drive amplitude. This way the two calculations can be performed
        simultaneously.

        A nice bonus is that both of the above mentioned calculations determine
        essentially independently whether a resonance occurs. In the first, it is
        deviation of the Floquet mode from the fitted displaced state. In the second,
        it is branch swapping that indicates a resonance, independent of any fit. Thus
        the two simulations can be used for cross validation of one another.

        We perform these simulations iteratively over the drive amplitudes as specified
        by fit_range_fraction. This is to allow for simulations stretching to large
        drive amplitudes, where the overlap with the bare eigenstate would fall below
        the threshold (due to ac Stark shift) even in the absence of any resonances.
        We thus use the fit from the previous range of drive amplitudes as our new bare
        state.
        """
        # Write the parameters to file and print them out
        write_to_h5(filepath, {}, self.param_dict())
        print(self)
        start_time = time.time()

        # initialize all arrays that will contain our data
        array_shape = (
            len(self.drive_parameters.omega_d_values),
            len(self.drive_parameters.drive_amplitudes),
            len(self.state_indices),
        )
        max_overlap_data = np.zeros(array_shape)
        # fit over the full range, using states identified by the fit over
        # intermediate ranges
        _displaced_state_overlaps = np.zeros(array_shape)
        floquet_mode_data = np.zeros((*array_shape, self.hilbert_dim), dtype=complex)
        avg_excitation = np.zeros(
            (
                len(self.drive_parameters.omega_d_values),
                len(self.drive_parameters.drive_amplitudes),
                self.hilbert_dim,
            )
        )
        quasienergies = np.zeros_like(avg_excitation)

        # for all omega_d, the bare states are identical at zero drive. We define
        # two sets of bare modes (prev_f_modes_arr and disp_coeffs_for_prev_amp)
        # because for the fit calculation, the bare modes are specified as fit
        # coefficients, whereas for the Blais calculation, the bare modes are specified
        # as actual kets.
        prev_f_modes_arr = np.tile(
            self.bare_state_array()[None, :, :],
            (len(self.drive_parameters.omega_d_values), 1, 1),
        )
        displaced_state = DisplacedStateFit(
            hilbert_dim=self.hilbert_dim,
            drive_parameters=self.drive_parameters,
            state_indices=self.state_indices,
            options=self.options,
        )
        previous_coefficients = np.array(
            [
                displaced_state.bare_state_coefficients(state_idx)
                for state_idx in self.state_indices
            ]
        )
        num_fit_ranges = int(np.ceil(1 / self.options.fit_range_fraction))
        num_amp_pts_per_range = int(
            np.floor(len(self.drive_parameters.drive_amplitudes) / num_fit_ranges)
        )
        for amp_range_idx in range(num_fit_ranges):
            print(f"calculating for amp_range_idx={amp_range_idx}")
            # edge case if range doesn't fit in neatly
            if amp_range_idx == num_fit_ranges - 1:
                amp_range_idx_final = len(self.drive_parameters.drive_amplitudes)
            else:
                amp_range_idx_final = (amp_range_idx + 1) * num_amp_pts_per_range
            amp_idxs = [amp_range_idx * num_amp_pts_per_range, amp_range_idx_final]
            # now perform floquet mode calculation for amp_range_idx
            # need to pass forward the floquet modes from the previous amp range
            # which allow us to identify floquet modes that may have been displaced
            # far from the origin
            output = self._floquet_main_for_amp_range(
                amp_idxs, displaced_state, previous_coefficients, prev_f_modes_arr
            )
            (
                max_overlap_data_for_range,
                floquet_mode_data_for_range,
                avg_excitation_for_range,
                quasienergies_for_range,
                prev_f_modes_arr,
            ) = output
            max_overlap_data = self._place_into(
                amp_idxs, max_overlap_data_for_range, max_overlap_data
            )
            floquet_mode_data = self._place_into(
                amp_idxs, floquet_mode_data_for_range, floquet_mode_data
            )
            avg_excitation = self._place_into(
                amp_idxs, avg_excitation_for_range, avg_excitation
            )
            quasienergies = self._place_into(
                amp_idxs, quasienergies_for_range, quasienergies
            )

            # ovlp_with_bare_states is used as a mask for the fit
            ovlp_with_bare_states = displaced_state.overlap_with_bare_states(
                amp_idxs[0], previous_coefficients, floquet_mode_data_for_range
            )
            omega_d_amp_data_slice = list(
                self.drive_parameters.omega_d_amp_params(amp_idxs)
            )
            # Compute the fitted 'ideal' displaced state, excluding those
            # floquet modes experiencing resonances.
            new_coefficients = displaced_state.displaced_states_fit(
                omega_d_amp_data_slice,
                ovlp_with_bare_states,
                floquet_mode_data_for_range,
            )
            # Compute overlap of floquet modes with ideal displaced state using this
            # new fit. We use this data as the mask for when we compute the coefficients
            # over the whole range. Note that we pass in floquet_mode_data as
            # opposed to the more restricted floquet_mode_data_for_range since we
            # use indexing methods inside of overlap_with_displaced_states, so its
            # easier to pass in the whole array.
            overlaps = displaced_state.overlap_with_displaced_states(
                amp_idxs, new_coefficients, floquet_mode_data
            )
            _displaced_state_overlaps = self._place_into(
                amp_idxs, overlaps, _displaced_state_overlaps
            )
            previous_coefficients = new_coefficients
        # the previously extracted coefficients were valid for the amplitude ranges
        # we asked for the fit over. Now armed with with correctly identified floquet
        # modes, we recompute these coefficients over the whole sea of floquet mode data
        # to get a plot that is free from numerical artifacts associated with
        # the fits being slightly different at the boundary of ranges
        amp_idxs = [0, len(self.drive_parameters.drive_amplitudes)]
        # In this case we utilize the previously computed overlaps of the floquet modes
        # with the displaced states (stored in _displaced_state_overlaps) to obtain the
        # mask with which we exclude some data from the fit (because we suspect they've
        # hit resonances).
        omega_d_amp_data_slice = list(
            self.drive_parameters.omega_d_amp_params(amp_idxs)
        )
        full_displaced_fit = displaced_state.displaced_states_fit(
            omega_d_amp_data_slice, _displaced_state_overlaps, floquet_mode_data
        )
        # try and be a little bit of a defensive programmer here, don't
        # just e.g. overwrite self.coefficient_matrix
        true_overlaps = displaced_state.overlap_with_displaced_states(
            amp_idxs, full_displaced_fit, floquet_mode_data
        )
        data_dict = {
            "max_overlap_data": max_overlap_data,
            "fit_data": full_displaced_fit,
            "displaced_state_overlaps": true_overlaps,
            "_displaced_state_overlaps": _displaced_state_overlaps,
            "quasienergies": quasienergies,
            "avg_excitation": avg_excitation,
        }
        if self.options.save_floquet_mode_data:
            data_dict["floquet_mode_data"] = floquet_mode_data
        print(f"finished in {(time.time() - start_time) / 60} minutes")
        update_data_in_h5(filepath, data_dict)
        return data_dict

    @staticmethod
    def _place_into(
        amp_idxs: list, array_for_range: np.ndarray, overall_array: np.ndarray
    ) -> np.ndarray:
        overall_array[:, amp_idxs[0] : amp_idxs[1]] = array_for_range
        return overall_array

    def _floquet_main_for_amp_range(
        self,
        amp_idxs: list,
        displaced_state: DisplacedState,
        previous_coefficients: np.ndarray,
        prev_f_modes_arr: np.ndarray,
    ) -> tuple:
        """Run the floquet simulation over a specific amplitude range."""
        amp_range_vals = self.drive_parameters.drive_amplitudes[
            amp_idxs[0] : amp_idxs[1]
        ]

        def _run_floquet_and_calculate(
            omega_d: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            omega_d_idx = self.drive_parameters.omega_d_to_idx(omega_d)
            amps_for_omega_d = amp_range_vals[:, omega_d_idx]
            avg_excitation_arr = np.zeros((len(amps_for_omega_d), self.hilbert_dim))
            quasienergies_arr = np.zeros_like(avg_excitation_arr)
            modes_quasies_ovlps_arr = np.zeros(
                (len(amps_for_omega_d), len(self.state_indices), 1 + self.hilbert_dim),
                dtype=complex,
            )
            prev_f_modes_for_omega_d = prev_f_modes_arr[omega_d_idx]
            # for evaluating the displaced state (might be dangerous to evaluate
            # outside of fitted window)
            params_0 = (omega_d, amp_range_vals[0, omega_d_idx])
            for amp_idx, amp in enumerate(amps_for_omega_d):
                params = (omega_d, amp)
                f_modes_energies = self.run_one_floquet(params)
                modes_quasies_ovlps = self.calculate_modes_quasies_ovlps(
                    f_modes_energies, params_0, displaced_state, previous_coefficients
                )
                modes_quasies_ovlps_arr[amp_idx] = modes_quasies_ovlps
                avg_excitation, quasi_es, new_f_modes_arr = self._step_in_amp(
                    f_modes_energies, prev_f_modes_for_omega_d
                )
                avg_excitation_arr[amp_idx] = np.real(avg_excitation)
                quasienergies_arr[amp_idx] = np.real(quasi_es)
                prev_f_modes_for_omega_d = new_f_modes_arr
            return (
                modes_quasies_ovlps_arr,
                avg_excitation_arr,
                quasienergies_arr,
                prev_f_modes_for_omega_d,
            )

        floquet_data = list(
            parallel_map(
                self.options.num_cpus,
                _run_floquet_and_calculate,
                self.drive_parameters.omega_d_values,
            )
        )
        (
            all_modes_quasies_ovlps,
            all_avg_excitation,
            all_quasienergies,
            f_modes_last_amp,
        ) = list(zip(*floquet_data, strict=False))
        floquet_mode_array = np.array(all_modes_quasies_ovlps, dtype=complex).reshape(
            (
                len(self.drive_parameters.omega_d_values),
                len(amp_range_vals),
                len(self.state_indices),
                1 + self.hilbert_dim,
            )
        )
        f_modes_last_amp = np.array(f_modes_last_amp, dtype=complex).reshape(
            (
                len(self.drive_parameters.omega_d_values),
                self.hilbert_dim,
                self.hilbert_dim,
            )
        )
        all_avg_excitation = np.array(all_avg_excitation).reshape(
            (
                len(self.drive_parameters.omega_d_values),
                len(amp_range_vals),
                self.hilbert_dim,
            )
        )
        all_quasienergies = np.array(all_quasienergies).reshape(
            (
                len(self.drive_parameters.omega_d_values),
                len(amp_range_vals),
                self.hilbert_dim,
            )
        )
        max_overlap_data = np.abs(floquet_mode_array[..., 0])
        floquet_mode_data = floquet_mode_array[..., 1:]
        return (
            max_overlap_data,
            floquet_mode_data,
            all_avg_excitation,
            all_quasienergies,
            f_modes_last_amp,
        )
