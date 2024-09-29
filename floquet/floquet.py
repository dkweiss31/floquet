from __future__ import annotations

import ast
import time

import numpy as np
import qutip as qt

from .displaced_state import DisplacedState, DisplacedStateFit
from .drive_parameters import DriveParameters
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


class FloquetAnalysis:
    """Class containing methods for performing full Floquet analysis.

    This class should be instantiated by calling the function floquet_analysis().
    In most workflows, one needs only then to call then run() method which performs
    both the displaced state fit and the Blais branch analysis. For an example
    workflow, see the [transmon](../examples/transmon) tutorial.
    """
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

    def run(self, filepath: str | None = None) -> dict:
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
        if filepath is not None:
            write_to_h5(filepath, {}, self.param_dict())
        print(self)
        start_time = time.time()

        # initialize all arrays that will contain our data
        array_shape = (
            len(self.drive_parameters.omega_d_values),
            len(self.drive_parameters.drive_amplitudes),
            len(self.state_indices),
        )
        bare_state_overlaps = np.zeros(array_shape)
        # fit over the full range, using states identified by the fit over
        # intermediate ranges
        intermediate_displaced_state_overlaps = np.zeros(array_shape)
        floquet_modes = np.zeros((*array_shape, self.hilbert_dim), dtype=complex)
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
                bare_state_overlaps_for_range,
                floquet_modes_for_range,
                avg_excitation_for_range,
                quasienergies_for_range,
                prev_f_modes_arr,
            ) = output
            bare_state_overlaps = self._place_into(
                amp_idxs, bare_state_overlaps_for_range, bare_state_overlaps
            )
            floquet_modes = self._place_into(
                amp_idxs, floquet_modes_for_range, floquet_modes
            )
            avg_excitation = self._place_into(
                amp_idxs, avg_excitation_for_range, avg_excitation
            )
            quasienergies = self._place_into(
                amp_idxs, quasienergies_for_range, quasienergies
            )

            # ovlp_with_bare_states is used as a mask for the fit
            ovlp_with_bare_states = displaced_state.overlap_with_bare_states(
                amp_idxs[0], previous_coefficients, floquet_modes_for_range
            )
            omega_d_amp_slice = list(self.drive_parameters.omega_d_amp_params(amp_idxs))
            # Compute the fitted 'ideal' displaced state, excluding those
            # floquet modes experiencing resonances.
            new_coefficients = displaced_state.displaced_states_fit(
                omega_d_amp_slice, ovlp_with_bare_states, floquet_modes_for_range
            )
            # Compute overlap of floquet modes with ideal displaced state using this
            # new fit. We use this data as the mask for when we compute the coefficients
            # over the whole range. Note that we pass in floquet_modes as
            # opposed to the more restricted floquet_modes_for_range since we
            # use indexing methods inside of overlap_with_displaced_states, so its
            # easier to pass in the whole array.
            overlaps = displaced_state.overlap_with_displaced_states(
                amp_idxs, new_coefficients, floquet_modes
            )
            intermediate_displaced_state_overlaps = self._place_into(
                amp_idxs, overlaps, intermediate_displaced_state_overlaps
            )
            previous_coefficients = new_coefficients
        # the previously extracted coefficients were valid for the amplitude ranges
        # we asked for the fit over. Now armed with with correctly identified floquet
        # modes, we recompute these coefficients over the whole sea of floquet mode data
        # to get a plot that is free from numerical artifacts associated with
        # the fits being slightly different at the boundary of ranges
        amp_idxs = [0, len(self.drive_parameters.drive_amplitudes)]
        # In this case we utilize the previously computed overlaps of the floquet modes
        # with the displaced states (stored in intermediate_displaced_state_overlaps)
        # to obtain the mask with which we exclude some data from the fit (because we
        # suspect they've hit resonances).
        omega_d_amp_slice = list(self.drive_parameters.omega_d_amp_params(amp_idxs))
        full_displaced_fit = displaced_state.displaced_states_fit(
            omega_d_amp_slice, intermediate_displaced_state_overlaps, floquet_modes
        )
        # try and be a little bit of a defensive programmer here, don't
        # just e.g. overwrite self.coefficient_matrix
        true_overlaps = displaced_state.overlap_with_displaced_states(
            amp_idxs, full_displaced_fit, floquet_modes
        )
        data_dict = {
            "bare_state_overlaps": bare_state_overlaps,
            "fit_data": full_displaced_fit,
            "displaced_state_overlaps": true_overlaps,
            "intermediate_displaced_state_overlaps": intermediate_displaced_state_overlaps,  # noqa E501
            "quasienergies": quasienergies,
            "avg_excitation": avg_excitation,
        }
        if self.options.save_floquet_modes:
            data_dict["floquet_modes"] = floquet_modes
        print(f"finished in {(time.time() - start_time) / 60} minutes")
        if filepath is not None:
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
        bare_state_overlaps = np.abs(floquet_mode_array[..., 0])
        floquet_modes = floquet_mode_array[..., 1:]
        return (
            bare_state_overlaps,
            floquet_modes,
            all_avg_excitation,
            all_quasienergies,
            f_modes_last_amp,
        )
