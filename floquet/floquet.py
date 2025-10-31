from __future__ import annotations

import time

import numpy as np
import qutip as qt

from .displaced_state import DisplacedState, DisplacedStateFit
from .model import Model
from .options import Options
from .utils.file_io import Serializable
from .utils.parallel import parallel_map


class FloquetAnalysis(Serializable):
    """Perform a floquet analysis to identify nonlinear resonances.

    In most workflows, one needs only to call the run() method which performs
    both the displaced state fit and the Blais branch analysis. For an example
    workflow, see the [transmon](../examples/transmon) tutorial.

    Parameters:
        model: Class specifying the model, including the Hamiltonian, drive amplitudes,
            frequencies
        state_indices: State indices of interest. Defaults to [0, 1], indicating the two
            lowest-energy states.
        options: Options for the Floquet analysis.
            ??? info "Detailed `opt_options` API"
                - fit_range_fraction (`float`, default: 1.0): Fraction of the amplitude
                    range to sweep over before changing the definition of the bare state
                    to that of the itted state from the previous range. For instance if
                    fit_range_fraction=0.4, then the amplitude range is split up into
                    three chunks: the first 40% of the amplitude linspace, then from
                    40% -> 80%, then from 80% to the full range. For the first fraction
                    of amplitudes, they are compared to the bare eigenstates for
                    identification. For the second range, they are compared to the
                    fitted state from the first range. And so on. Defaults to 1.0,
                    indicating that no iteration is performed.
                - floquet_sampling_time_fraction (`float`, default: 0.0): What point of
                    the drive period we want to sample the Floquet modes. Defaults to
                    0.0, indicating the floquet modes at t=0*T where T is the drive
                    period.
                - fit_cutoff (`int`, default: 4): Cutoff for the fit polynomial of the
                    displaced state.
                - overlap_cutoff (`float`, default: 0.8): Cutoff for fitting overlaps.
                    Floquet modes with overlap with the "bare" state below this cutoff
                    are not included in the fit (as they may be experiencing a
                    resonance).
                - nsteps (`int`, default: 30_000): QuTiP integration parameter, number
                    of steps the solver can take.
                - num_cpus (`int`, default: 1): Number of cpus to use in parallel
                    computation of Floquet modes over the different values of
                    omega_d, amp.
                - save_floquet_modes (`bool`, default: False): Indicating whether to
                    save the extracted Floquet modes themselves. Such data is often
                    unnecessary and requires a fair amount of storage, so the default is
                    False.
        init_data_to_save: Initial parameter metadata to save to file. Defaults to None.
    """

    def __init__(
        self,
        model: Model,
        state_indices: list | None = None,
        options: Options = Options(),  # noqa B008
        init_data_to_save: dict | None = None,
    ):
        if state_indices is None:
            state_indices = [0, 1]
        self.model = model
        self.state_indices = state_indices
        self.options = options
        self.init_data_to_save = init_data_to_save
        self.hilbert_dim = model.H0.shape[0]

    def __str__(self) -> str:
        return "Running floquet simulation with parameters: \n" + super().__str__()

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
        fbasis = qt.FloquetBasis(
            self.model.hamiltonian(omega_d_amp),
            T,
            options={"nsteps": self.options.nsteps},  # type: ignore
        )
        f_modes_0 = fbasis.mode(0.0)
        f_energies = fbasis.e_quasi
        sampling_time = self.options.floquet_sampling_time_fraction * T % T
        f_modes_t = fbasis.mode(sampling_time) if sampling_time != 0.0 else f_modes_0
        return f_modes_t, f_energies

    def identify_floquet_modes(
        self,
        f_modes_energies: tuple[np.ndarray, qt.Qobj],
        params_0: tuple[float, float],
        displaced_state: DisplacedState,
        previous_coefficients: np.ndarray,
    ) -> np.ndarray:
        """Return floquet modes with largest overlap with ideal displaced state.

        Also return that overlap value.

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
            [f_modes_0[idx].data.to_array()[:, 0] for idx in range(self.hilbert_dim)],
            dtype=complex,
        ).T
        # return overlap and floquet mode
        ovlps_and_modes = np.zeros(
            (len(self.state_indices), 1 + self.hilbert_dim), dtype=complex
        )
        # TODO: refactor using overlap_with_bare_states method?
        ideal_displaced_state_array = np.array(
            [
                displaced_state.displaced_state(
                    *params_0, state_idx, previous_coefficients[array_idx]
                )
                .dag()
                .data.to_array()[0]
                for array_idx, state_idx in enumerate(self.state_indices)
            ]
        )
        overlaps = np.einsum("ij,jk->ik", ideal_displaced_state_array, f_modes_cols)
        # take the argmax along k
        f_idxs = np.argmax(np.abs(overlaps), axis=1)
        for array_idx, _state_idx in enumerate(self.state_indices):
            f_idx = f_idxs[array_idx]
            bare_state_overlap = overlaps[array_idx, f_idx]
            ovlps_and_modes[array_idx, 0] = bare_state_overlap
            ovlps_and_modes[array_idx, 1:] = (
                np.sign(bare_state_overlap) * f_modes_cols[:, f_idx]
            )
        return ovlps_and_modes

    def bare_state_array(self) -> np.ndarray:
        """Return array of bare states.

        Used to specify initial bare states for the Blais branch analysis.
        """
        return np.squeeze(
            np.array(
                [
                    qt.basis(self.hilbert_dim, idx).data.to_array()
                    for idx in range(self.hilbert_dim)
                ]
            )
        )

    def _step_in_amp(
        self, f_modes_energies: tuple[qt.Qobj, np.ndarray], prev_f_modes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform Blais branch analysis.

        Gorgeous in its simplicity. Simply calculate overlaps of new floquet modes with
        those from the previous amplitude step, and order the modes accordingly. So
        ordered, compute the mean excitation number, yielding our branches.
        """
        f_modes_0, f_energies_0 = f_modes_energies
        f_modes_0 = np.squeeze([f_mode.data.to_array() for f_mode in f_modes_0])
        all_overlaps = np.abs(np.einsum("ij,kj->ik", np.conj(prev_f_modes), f_modes_0))
        # assume that prev_f_modes_arr have been previously sorted. Question
        # is which k index has max overlap?
        max_idxs = np.argmax(all_overlaps, axis=1)
        f_modes_ordered = f_modes_0[max_idxs]
        avg_excitation = self._calculate_mean_excitation(f_modes_ordered)
        return avg_excitation, f_energies_0[max_idxs], f_modes_ordered

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
        print(self)
        start_time = time.time()

        # initialize all arrays that will contain our data
        array_shape = (
            len(self.model.omega_d_values),
            len(self.model.drive_amplitudes),
            len(self.state_indices),
        )
        bare_state_overlaps = np.zeros(array_shape)
        # fit over the full range, using states identified by the fit over
        # intermediate ranges
        intermediate_displaced_state_overlaps = np.zeros(array_shape)
        floquet_modes = np.zeros((*array_shape, self.hilbert_dim), dtype=complex)
        avg_excitation = np.zeros(
            (
                len(self.model.omega_d_values),
                len(self.model.drive_amplitudes),
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
            self.bare_state_array()[None, :, :], (len(self.model.omega_d_values), 1, 1)
        )
        displaced_state = DisplacedStateFit(
            hilbert_dim=self.hilbert_dim,
            model=self.model,
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
            np.floor(len(self.model.drive_amplitudes) / num_fit_ranges)
        )
        for amp_range_idx in range(num_fit_ranges):
            print(f"calculating for amp_range_idx={amp_range_idx}")
            # edge case if range doesn't fit in neatly
            if amp_range_idx == num_fit_ranges - 1:
                amp_range_idx_final = len(self.model.drive_amplitudes)
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
            omega_d_amp_slice = list(self.model.omega_d_amp_params(amp_idxs))
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
        # The previously extracted coefficients were valid for the amplitude ranges
        # we asked for the fit over. Now armed with with correctly identified floquet
        # modes, we recompute these coefficients over the whole sea of floquet mode data
        # to get a plot that is free from numerical artifacts associated with
        # the fits being slightly different at the boundary of ranges. We utilize the
        # previously computed overlaps of the floquet modes with the displaced states
        # (stored in intermediate_displaced_state_overlaps) to obtain the mask with
        # which we exclude some data from the fit (because we suspect they've hit
        # resonances).
        amp_idxs = [0, len(self.model.drive_amplitudes)]
        omega_d_amp_slice = list(self.model.omega_d_amp_params(amp_idxs))
        full_displaced_fit = displaced_state.displaced_states_fit(
            omega_d_amp_slice, intermediate_displaced_state_overlaps, floquet_modes
        )
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
            self.write_to_file(filepath, data_dict)
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
        amp_range_vals = self.model.drive_amplitudes[amp_idxs[0] : amp_idxs[1]]

        def _run_floquet_and_calculate(
            omega_d: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            omega_d_idx = self.model.omega_d_to_idx(omega_d)
            amps_for_omega_d = amp_range_vals[:, omega_d_idx]
            avg_excitation_arr = np.zeros((len(amps_for_omega_d), self.hilbert_dim))
            quasienergies_arr = np.zeros_like(avg_excitation_arr)
            ovlps_and_modes_arr = np.zeros(
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
                ovlps_and_modes = self.identify_floquet_modes(
                    f_modes_energies, params_0, displaced_state, previous_coefficients
                )
                ovlps_and_modes_arr[amp_idx] = ovlps_and_modes
                avg_excitation, quasi_es, new_f_modes_arr = self._step_in_amp(
                    f_modes_energies, prev_f_modes_for_omega_d
                )
                avg_excitation_arr[amp_idx] = np.real(avg_excitation)
                quasienergies_arr[amp_idx] = np.real(quasi_es)
                prev_f_modes_for_omega_d = new_f_modes_arr
            return (
                ovlps_and_modes_arr,
                avg_excitation_arr,
                quasienergies_arr,
                prev_f_modes_for_omega_d,
            )

        floquet_data = list(
            parallel_map(
                self.options.num_cpus,
                _run_floquet_and_calculate,
                self.model.omega_d_values,
            )
        )
        (
            all_modes_quasies_ovlps,
            all_avg_excitation,
            all_quasienergies,
            f_modes_last_amp,
        ) = list(zip(*floquet_data, strict=True))
        floquet_mode_array = np.array(all_modes_quasies_ovlps, dtype=complex).reshape(
            (
                len(self.model.omega_d_values),
                len(amp_range_vals),
                len(self.state_indices),
                1 + self.hilbert_dim,
            )
        )
        f_modes_last_amp = np.array(f_modes_last_amp, dtype=complex).reshape(
            (len(self.model.omega_d_values), self.hilbert_dim, self.hilbert_dim)
        )
        all_avg_excitation = np.array(all_avg_excitation).reshape(
            (len(self.model.omega_d_values), len(amp_range_vals), self.hilbert_dim)
        )
        all_quasienergies = np.array(all_quasienergies).reshape(
            (len(self.model.omega_d_values), len(amp_range_vals), self.hilbert_dim)
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
