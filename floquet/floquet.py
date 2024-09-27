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
    state_indices: list | None = None,
    omega_d_linspace: np.ndarray | list = 2.0 * np.pi * np.linspace(6.9, 13, 50),  # noqa B008
    amp_linspace: np.ndarray | list = 2.0 * np.pi * np.linspace(0.0, 0.2, 51),  # noqa B008
    options: Options = Options(),  # noqa B008
    init_data_to_save: dict | None = None,
) -> FloquetAnalysis:
    """Perform a floquet analysis to identify nonlinear resonances.

    Arguments:
        H0: qt.Qobj | np.ndarray | list
            Drift Hamiltonian (ideally diagonal)
        H1: qt.Qobj | np.ndarray | list
            drive operator
        state_indices: list
            state indices of interest
        omega_d_linspace: ndarray | list
            drive frequencies to scan over
        amp_linspace: ndarray | list
            amp values to scan over. Can be one dimensional in which case
            these amplitudes are used for all omega_d, or it can be two dimensional
            in which case the first dimension are the amplitudes to scan over
            and the second are the amplitudes for respective drive frequencies
        options: Options
            Options for the Floquet analysis.
        init_data_to_save: dict | None
            Initial parameter metadata to save to file. Defaults to None.
    """
    if state_indices is None:
        state_indices = [0, 1]
    if not isinstance(H0, qt.Qobj):
        H0 = qt.Qobj(np.array(H0, dtype=complex))
    if not isinstance(H1, qt.Qobj):
        H1 = qt.Qobj(np.array(H1, dtype=complex))
    if isinstance(omega_d_linspace, list):
        omega_d_linspace = np.array(omega_d_linspace)
    if isinstance(amp_linspace, list):
        amp_linspace = np.array(amp_linspace)
    if len(amp_linspace.shape) == 1:
        amp_linspace = np.tile(amp_linspace, (len(omega_d_linspace), 1)).T
    else:
        assert len(amp_linspace.shape) == 2
        assert amp_linspace.shape[1] == len(omega_d_linspace)
    return FloquetAnalysis(
        H0,
        H1,
        state_indices,
        omega_d_linspace,
        amp_linspace,
        options,
        init_data_to_save=init_data_to_save,
    )


def floquet_analysis_from_file(filepath: str) -> FloquetAnalysis:
    _, param_dict = extract_info_from_h5(filepath)
    floquet_init = ast.literal_eval(param_dict['floquet_analysis_init'])
    return floquet_analysis(
        floquet_init['H0'],
        floquet_init['H1'],
        state_indices=floquet_init['state_indices'],
        omega_d_linspace=floquet_init['omega_d_linspace'],
        amp_linspace=floquet_init['amp_linspace'],
        options=Options(**floquet_init['options']),
        init_data_to_save=floquet_init['init_data_to_save'],
    )


class FloquetAnalysis:
    def __init__(
        self,
        H0: qt.Qobj,
        H1: qt.Qobj,
        state_indices: list,
        omega_d_linspace: np.ndarray,
        amp_linspace: np.ndarray,
        options: Options,
        init_data_to_save: dict | None = None,
    ):
        self.H0 = H0
        self.H1 = H1
        self.state_indices = state_indices
        self.omega_d_linspace = omega_d_linspace
        self.amp_linspace = amp_linspace
        self.options = options
        self.init_data_to_save = init_data_to_save
        # Save in _init_attrs for later re-initialization. Everything added to self
        # after this is a derived quantity
        self._init_attrs = set(self.__dict__.keys())
        self.num_states = H0.shape[0]
        self.exponent_pair_idx_map = self._create_exponent_pair_idx_map()
        self._num_fit_ranges = int(np.ceil(1 / options.fit_range_fraction))
        self._num_amp_pts_per_range = int(
            np.floor(len(self.amp_linspace) / self._num_fit_ranges)
        )
        array_shape = (
            len(self.omega_d_linspace),
            len(self.amp_linspace),
            len(self.state_indices),
        )
        self.max_overlap_data = np.zeros(array_shape)
        self.quasienergies = np.zeros(array_shape)
        # fit over the full range, using states identified by the fit over
        # intermediate ranges
        self.coefficient_matrix = np.zeros(
            (len(self.state_indices), self.num_states, len(self.exponent_pair_idx_map)),
            dtype=complex,
        )
        self.displaced_state_overlaps = np.zeros(array_shape)
        self._displaced_state_overlaps = np.zeros(array_shape)
        self.floquet_mode_idxs = np.zeros(array_shape, dtype=int)
        self.floquet_mode_data = np.zeros(
            (*array_shape, self.num_states), dtype=complex
        )
        self.avg_excitation = np.zeros(
            (len(self.omega_d_linspace), len(self.amp_linspace), self.num_states)
        )
        self.all_quasienergies = np.zeros_like(self.avg_excitation)

    def _get_initdata(self) -> dict:
        """Collect all attributes for writing to file."""
        init_dict = {k: v for k, v in self.__dict__.items() if k in self._init_attrs}
        new_init_dict = {}
        for k, v in init_dict.items():
            if isinstance(v, qt.Qobj):
                new_init_dict[k] = v.data.toarray().tolist()
            elif isinstance(v, np.ndarray):
                new_init_dict[k] = v.tolist()
            elif isinstance(v, Options):
                new_init_dict['options'] = vars(v)
            else:
                new_init_dict[k] = v
        return new_init_dict

    def param_dict(self) -> dict:
        if self.init_data_to_save is None:
            self.init_data_to_save = {}
        param_dict = vars(self.options) | self.init_data_to_save
        return param_dict | {
            'state_indices': self.state_indices,
            'omega_d_linspace': self.omega_d_linspace,
            'amp_linspace': self.amp_linspace,
            'num_states': self.num_states,
            'num_fit_ranges': self._num_fit_ranges,
            'num_amp_pts_per_range': self._num_amp_pts_per_range,
            'exp_pair_map': self.exponent_pair_idx_map,
            'floquet_analysis_init': self._get_initdata(),
        }

    def __str__(self):
        params = self.param_dict()
        params.pop('floquet_analysis_init')
        parts_str = '\n'.join(f'{k}: {v}' for k, v in params.items() if v is not None)
        return f'Running floquet simulation with parameters: \n' + parts_str

    def run(self, filepath: str = 'tmp.h5py') -> None:
        """Run the whole floquet simulation."""
        write_to_h5(filepath, {}, self.param_dict())
        print(self)
        start_time = time.time()
        self.floquet_main(filepath=filepath)
        print(f'finished floquet main in {(time.time()-start_time)/60} minutes')
        update_data_in_h5(filepath, self.assemble_data_dict())

    def assemble_data_dict(self) -> dict:
        """Collect all computed data in preparation for writing to file."""
        data_dict = {
            'max_overlap_data': self.max_overlap_data,
            'floquet_mode_idxs': self.floquet_mode_idxs,
            'fit_data': self.coefficient_matrix,
            'quasienergies': self.quasienergies,
            'displaced_state_overlaps': self.displaced_state_overlaps,
            '_displaced_state_overlaps': self._displaced_state_overlaps,
            'all_quasienergies': self.all_quasienergies,
            'avg_excitation': self.avg_excitation,
        }
        if self.options.save_floquet_mode_data:
            data_dict['floquet_mode_data'] = self.floquet_mode_data
        return data_dict

    def hamiltonian(self, params: tuple[float, float]) -> list[qt.Qobj]:
        """Return the Hamiltonian we actually simulate."""
        omega_d, amp = params
        return [self.H0, [amp * self.H1, lambda t, _: np.cos(omega_d * t)]]

    def run_one_floquet(
        self, params: tuple[float, float]
    ) -> tuple[np.ndarray, qt.Qobj]:
        """Run one instance of the problem for a pair of drive frequency and amp.

        Returns floquet modes as numpy column vectors, as well as the quasienergies.
        Parameters.
        ----------
        params: tuple
            pair of drive frequency and amp
        """
        omega_d, amp = params
        T = 2.0 * np.pi / omega_d
        f_modes_0, f_energies_0 = qt.floquet_modes(
            self.hamiltonian(params),  # type: ignore
            T,
            options=qt.Options(nsteps=self.options.nsteps),
        )
        sampling_time = self.options.floquet_sampling_time_fraction * T % T
        if sampling_time != 0:
            f_modes_t = qt.floquet_modes_t(
                f_modes_0,
                f_energies_0,
                sampling_time,
                self.hamiltonian(params),  # type: ignore
                T,
                options=qt.Options(nsteps=self.options.nsteps),
            )
        else:
            f_modes_t = f_modes_0
        return f_modes_t, f_energies_0

    def _calculate_modes_quasies_ovlps(
        self,
        f_modes_energies: tuple[np.ndarray, qt.Qobj],
        params_0: tuple[float, float],
        disp_coeffs: np.ndarray,
    ) -> np.ndarray:
        """Return overlaps with "ideal" bare state at a given pair of (omega_d, amp).

        Parameters.
        ----------
        f_modes_energies: tuple
            output of self.run_one_floquet(params)
        params_0: tuple
            (omega_d_0, amp_0) to use for displaced fit
        disp_coeffs: ndarray
            matrix of coefficients for the displaced state
        """
        f_modes_0, f_energies_0 = f_modes_energies
        # construct column vectors to compute overlaps
        f_modes_cols = np.array(
            [f_modes_0[idx].data.toarray()[:, 0] for idx in range(self.num_states)],
            dtype=complex,
        ).T
        # return overlap and floquet mode
        modes_quasies_ovlps = np.zeros(
            (len(self.state_indices), 3 + self.num_states), dtype=complex
        )
        ideal_displaced_state_array = np.array(
            [
                self.displaced_state(*params_0, disp_coeffs[array_idx], state_idx)
                .dag()
                .data.toarray()[0]
                for array_idx, state_idx in enumerate(self.state_indices)
            ]
        )
        overlaps = np.einsum('ij,jk->ik', ideal_displaced_state_array, f_modes_cols)
        # take the argmax along k
        f_idxs = np.argmax(np.abs(overlaps), axis=1)
        for array_idx, _state_idx in enumerate(self.state_indices):
            f_idx = f_idxs[array_idx]
            bare_state_overlap = overlaps[array_idx, f_idx]
            modes_quasies_ovlps[array_idx, 0] = bare_state_overlap
            modes_quasies_ovlps[array_idx, 1] = f_idx
            modes_quasies_ovlps[array_idx, 2] = f_energies_0[f_idx]
            modes_quasies_ovlps[array_idx, 3:] = (
                np.sign(bare_state_overlap) * f_modes_cols[:, f_idx]
            )
        return modes_quasies_ovlps

    def bare_state_array(self) -> np.ndarray:
        """Return array of bare states.

        Used to specify initial bare states for the Blais branch analysis.
        """
        return np.squeeze(
            np.array([qt.basis(self.num_states, idx) for idx in range(self.num_states)])
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
        all_overlaps = np.abs(np.einsum('ij,kj->ik', np.conj(prev_f_modes), f_modes_0))
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
        overlaps_sq = np.abs(np.einsum('ij,kj->ik', bare_states, f_modes_ordered)) ** 2
        # sum over bare excitations weighted by excitation number
        return np.einsum('ik,i->k', overlaps_sq, np.arange(0, self.num_states))

    def _single_overlap_with_displaced_state(
        self, omega_d_amp: tuple[float, float], disp_coeffs_for_new_amp: np.ndarray
    ) -> np.ndarray:
        """Calculate overlap of floquet mode with 'ideal' displaced state.

        This is done here for a given omega_d, amp pair.
        """
        overlap = np.zeros(len(self.state_indices))
        omega_d, amp = omega_d_amp
        for array_idx, state_idx in enumerate(self.state_indices):
            floquet_data = self.floquet_mode_data[
                self.omega_d_to_idx(omega_d), self.amp_to_idx(amp, omega_d), array_idx
            ]
            disp_state = self.displaced_state(
                omega_d, amp, disp_coeffs_for_new_amp[array_idx], state_idx=state_idx
            ).dag()
            overlap[array_idx] = np.abs(disp_state.data.toarray()[0] @ floquet_data)
        return overlap

    def floquet_main(self, filepath: str = 'tmp.h5py') -> None:
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
        # for all omega_d, the bare states are identical at zero drive. We define
        # two sets of bare modes (prev_f_modes_arr and disp_coeffs_for_prev_amp)
        # because for the fit calculation, the bare modes are specified as fit
        # coefficients, whereas for the Blais calculation, the bare modes are specified
        # as actual kets.
        prev_f_modes_arr = np.tile(
            self.bare_state_array()[None, :, :], (len(self.omega_d_linspace), 1, 1)
        )
        disp_coeffs_for_prev_amp = np.array(
            [
                self.bare_state_coefficients(state_idx)
                for state_idx in self.state_indices
            ]
        )
        for amp_range_idx in range(self._num_fit_ranges):
            print(f'calculating for amp_range_idx={amp_range_idx}')
            # edge case if range doesn't fit in neatly
            if amp_range_idx == self._num_fit_ranges - 1:
                amp_range_idx_final = len(self.amp_linspace)
            else:
                amp_range_idx_final = (amp_range_idx + 1) * self._num_amp_pts_per_range
            amp_idxs = [
                amp_range_idx * self._num_amp_pts_per_range,
                amp_range_idx_final,
            ]
            # now perform floquet mode calculation for amp_range_idx
            # need to pass forward the floquet modes from the previous amp range
            # which allow us to identify floquet modes that may have been displaced
            # far from the origin
            prev_f_modes_arr = self._floquet_main_for_amp_range(
                amp_idxs, disp_coeffs_for_prev_amp, prev_f_modes_arr
            )
            disp_coeffs_for_prev_amp = self._displaced_states_fit(
                amp_idxs, disp_coeffs_for_prev_amp
            )
            overlaps = self._overlap_with_displaced_states(
                amp_idxs, disp_coeffs_for_prev_amp
            )
            # save the extracted overlaps for later use when fitting over the whole
            # shebang
            self._displaced_state_overlaps[:, amp_idxs[0]: amp_idxs[1], :] = overlaps
            update_data_in_h5(filepath, self.assemble_data_dict())
        # the previously extracted coefficients were valid for the amplitude ranges
        # we asked for the fit over. Now armed with with correctly identified floquet
        # modes, we recompute these coefficients over the whole sea of floquet mode data
        # to get a plot that is free from numerical artifacts associated with
        # the fits being slightly different at the boundary of ranges
        full_displaced_fit = self._displaced_states_fit()
        # try and be a little bit of a defensive programmer here, don't
        # just e.g. overwrite self.coefficient_matrix
        self.coefficient_matrix[:, :, :] = full_displaced_fit
        true_overlaps = self._overlap_with_displaced_states(
            [0, len(self.amp_linspace)], full_displaced_fit
        )
        self.displaced_state_overlaps[:, :, :] = true_overlaps

    def _floquet_main_for_amp_range(
        self,
        amp_idxs: list,
        disp_coeffs_for_prev_amp: np.ndarray,
        prev_f_modes_arr: np.ndarray,
    ) -> np.ndarray:
        """Run the floquet simulation over a specific amplitude range."""
        amp_range_vals = self.amp_linspace[amp_idxs[0]: amp_idxs[1]]

        def _run_floquet_and_calculate(
            omega_d: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            omega_d_idx = self.omega_d_to_idx(omega_d)
            amps_for_omega_d = amp_range_vals[:, omega_d_idx]
            avg_excitation_arr = np.zeros((len(amps_for_omega_d), self.num_states))
            quasienergies_arr = np.zeros_like(avg_excitation_arr)
            modes_quasies_ovlps_arr = np.zeros(
                (len(amps_for_omega_d), len(self.state_indices), 3 + self.num_states),
                dtype=complex,
            )
            prev_f_modes_for_omega_d = prev_f_modes_arr[omega_d_idx]
            # for evaluating the displaced state (might be dangerous to evaluate
            # outside of fitted window)
            params_0 = (omega_d, amp_range_vals[0, omega_d_idx])
            for amp_idx, amp in enumerate(amps_for_omega_d):
                params = (omega_d, amp)
                f_modes_energies = self.run_one_floquet(params)
                modes_quasies_ovlps = self._calculate_modes_quasies_ovlps(
                    f_modes_energies, params_0, disp_coeffs_for_prev_amp
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
                self.options.num_cpus, _run_floquet_and_calculate, self.omega_d_linspace
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
                len(self.omega_d_linspace),
                len(amp_range_vals),
                len(self.state_indices),
                3 + self.num_states,
            )
        )
        f_modes_last_amp = np.array(f_modes_last_amp, dtype=complex).reshape(
            (len(self.omega_d_linspace), self.num_states, self.num_states)
        )
        self.max_overlap_data[:, amp_idxs[0]: amp_idxs[1]] = np.real(
            floquet_mode_array[..., 0]
        )
        self.floquet_mode_idxs[:, amp_idxs[0]: amp_idxs[1]] = np.real(
            floquet_mode_array[..., 1]
        )
        self.quasienergies[:, amp_idxs[0]: amp_idxs[1]] = np.real(
            floquet_mode_array[..., 2]
        )
        self.floquet_mode_data[:, amp_idxs[0]: amp_idxs[1]] = floquet_mode_array[
            ..., 3:
        ]
        self.avg_excitation[:, amp_idxs[0]: amp_idxs[1]] = np.array(
            all_avg_excitation
        ).reshape((len(self.omega_d_linspace), len(amp_range_vals), self.num_states))
        self.all_quasienergies[:, amp_idxs[0]: amp_idxs[1]] = np.array(
            all_quasienergies
        ).reshape((len(self.omega_d_linspace), len(amp_range_vals), self.num_states))
        return f_modes_last_amp

    def _omega_d_amp_params(self, amp_idxs: list | None = None) -> itertools.chain:
        """Return ordered chain object of the specified omega_d and amplitude values."""
        if amp_idxs is None:
            amp_range_vals = self.amp_linspace
        else:
            amp_range_vals = self.amp_linspace[amp_idxs[0]: amp_idxs[1]]
        _omega_d_amp_params = [
            product([omega_d], amp_vals)
            for omega_d, amp_vals in zip(
                self.omega_d_linspace, amp_range_vals.T, strict=False
            )
        ]
        return chain(*_omega_d_amp_params)

    def _overlap_with_displaced_states(
        self, amp_idxs: list, disp_coeffs_for_new_amp: np.ndarray
    ) -> np.ndarray:
        """Calculate overlap of floquet modes with 'ideal' displaced states.

        This is done here for a specific amplitude range.
        """

        def run_overlap_displaced(omega_d_amp: tuple[float, float]) -> np.ndarray:
            return self._single_overlap_with_displaced_state(
                omega_d_amp, disp_coeffs_for_new_amp
            )

        omega_d_amp_params = self._omega_d_amp_params(amp_idxs)
        amp_range_vals = self.amp_linspace[amp_idxs[0]: amp_idxs[1]]
        result = list(
            parallel_map(
                self.options.num_cpus, run_overlap_displaced, omega_d_amp_params
            )
        )
        return np.array(result).reshape(
            (len(self.omega_d_linspace), len(amp_range_vals), len(self.state_indices))
        )

    def _displaced_states_fit(
        self, amp_idxs: list | None = None, disp_coeffs_for_prev_amp: np.ndarray = None
    ) -> np.ndarray:
        """Loop over all states and perform the fit for a given amplitude range.

        If amp_idxs and disp_coeffs_for_prev_amp are None, then this indicates that we
        are on the final pass and we should perform the fit over the whole range of
        amplitudes. In this case we utilize the previously computed overlaps of the
        floquet modes with the displaced states to obtain the mask with which we exclude
        some data from the fit (because we suspect they've hit resonances).
        """

        def fit_for_state(array_state_idx: tuple[int, int]) -> np.ndarray:
            return self._displaced_states_fit_for_amp_and_state(
                amp_idxs, disp_coeffs_for_prev_amp, array_state_idx
            )

        array_idxs = np.arange(len(self.state_indices))
        array_state_idxs = zip(array_idxs, self.state_indices, strict=False)
        fit_data = list(
            parallel_map(self.options.num_cpus, fit_for_state, array_state_idxs)
        )
        return np.array(fit_data, dtype=complex).reshape(
            (len(self.state_indices), self.num_states, len(self.exponent_pair_idx_map))
        )

    def _ravel_and_filter_params(
        self, mask: np.ndarray, omega_d_amp_params: list
    ) -> list:
        mask = mask.ravel()
        return [
            omega_d_amp_params[i]
            for i in range(len(mask))
            if np.abs(mask[i]) > self.options.overlap_cutoff
        ]

    def _ravel_and_filter_mode_data(
        self, mask: np.ndarray, floquet_data: np.ndarray, state_idx_component: int
    ) -> np.ndarray:
        mask = mask.ravel()
        floquet_idx_data_bare_component = floquet_data[
            :, :, state_idx_component
        ].ravel()
        # only fit states that we think haven't run into
        # a nonlinear transition (same for omega_d_amp_filtered above)
        return floquet_idx_data_bare_component[
            np.abs(mask) > self.options.overlap_cutoff
        ]

    def _displaced_states_fit_for_amp_and_state(
        self,
        amp_idxs: list | None = None,
        disp_coeffs_for_prev_amp: np.ndarray | None = None,
        array_state_idx: tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """Find the displaced states.

        This is done here for all frequencies, a given amplitude window and a given
        state.
        """
        # For the bare state, we look at the amplitude at the leading edge of the
        # window (that is, the smallest amplitude). This is the most natural choice,
        # as it is most analogous to what is done in the first window when the overlap
        # is computed against bare eigenstates (that obviously don't have amplitude
        # dependence). Moreover, the fit coefficients for the previous window by
        # definition were obtained in a window that does not include the one we are
        # currently investigating. Asking for the state for amplitude values outside of
        # the fit window should be done at your own peril.
        array_idx, state_idx = array_state_idx
        if amp_idxs is None:
            floquet_idx_data = self.floquet_mode_data[:, :, array_idx, :]
        else:
            floquet_idx_data = self.floquet_mode_data[
                :, amp_idxs[0]: amp_idxs[1], array_idx, :
            ]
        if disp_coeffs_for_prev_amp is None:
            # this means we are on the final lap, and want to compare with previously
            # computed overlaps
            ovlp_with_bare_state = self._displaced_state_overlaps[:, :, array_idx]
        else:

            def _compute_bare_state(omega_d: float) -> np.ndarray:
                omega_d_idx = self.omega_d_to_idx(omega_d)
                return self.displaced_state(
                    omega_d,
                    self.amp_linspace[amp_idxs[0], omega_d_idx],
                    disp_coeffs_for_prev_amp[array_idx],
                    state_idx,
                ).full()[:, 0]

            bare_states = np.array(
                [_compute_bare_state(omega_d) for omega_d in self.omega_d_linspace],
                dtype=complex,
            )
            # bare states may differ as a function of omega_d, hence the bare states
            # have an index of i that we don't sum over
            # indices are i: omega_d, j: amp, k: components of state
            # this serves as the mask that is passed to _disp_coeffs_fit
            ovlp_with_bare_state = np.abs(
                np.einsum('ijk,ik->ij', floquet_idx_data, np.conj(bare_states))
            )
        omega_d_amp_data_slice = list(self._omega_d_amp_params(amp_idxs))
        omega_d_amp_filtered = self._ravel_and_filter_params(
            ovlp_with_bare_state, omega_d_amp_data_slice
        )
        if len(omega_d_amp_filtered) < len(self.exponent_pair_idx_map):
            warnings.warn(
                'Not enough data points to fit. Returning previous fit', stacklevel=3
            )
            return disp_coeffs_for_prev_amp
        num_coeffs = len(self.exponent_pair_idx_map)
        coefficient_matrix_for_amp_and_state = np.zeros(
            (self.num_states, num_coeffs), dtype=complex
        )
        for state_idx_component in range(self.num_states):
            floquet_component_filtered = self._ravel_and_filter_mode_data(
                ovlp_with_bare_state, floquet_idx_data, state_idx_component
            )
            bare_same = state_idx_component == state_idx
            bare_component_fit = self._disp_coeffs_fit(
                omega_d_amp_filtered, floquet_component_filtered, bare_same
            )
            coefficient_matrix_for_amp_and_state[state_idx_component, :] = (
                bare_component_fit
            )
        return coefficient_matrix_for_amp_and_state

    def _disp_coeffs_fit(
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
        popt_r = self._fit_params(
            omega_d_amp_filtered, np.real(floquet_component_filtered), p0, bare_same
        )
        popt_i = self._fit_params(
            omega_d_amp_filtered,
            np.imag(floquet_component_filtered),
            p0,
            False,  # for the imaginary part, constant term should always be zero
        )
        return popt_r + 1j * popt_i

    def bare_state_coefficients(self, state_idx: int) -> np.ndarray:
        """For bare state only component is itself."""
        coefficient_matrix_for_amp_and_state = np.zeros(
            (self.num_states, len(self.exponent_pair_idx_map)), dtype=complex
        )
        coefficient_matrix_for_amp_and_state[state_idx, 0] = 1.0
        return coefficient_matrix_for_amp_and_state

    def floquet_mode(self, omega_d: float, amp: float, array_idx: int) -> np.ndarray:
        """Helper function for extracting the floquet mode."""
        omega_d_idx = self.omega_d_to_idx(omega_d)
        amp_idx = self.amp_to_idx(amp, omega_d)
        return self.floquet_mode_data[omega_d_idx, amp_idx, array_idx]

    def _fit_params(
        self, XYdata: list, Zdata: np.ndarray, p0: tuple | np.ndarray, bare_same: bool
    ) -> np.ndarray:
        poly_fit = functools.partial(self._polynomial_fit, bare_same=bare_same)
        try:
            popt, pcov = sp.optimize.curve_fit(poly_fit, XYdata, Zdata, p0=p0)
        except RuntimeError:
            warnings.warn(
                'fit failed for a bare component, returning zeros for the fit',
                stacklevel=3,
            )
            popt = np.zeros(len(p0))
        return popt

    def _create_exponent_pair_idx_map(self) -> dict:
        """Create dictionary of terms in polynomial that we fit.

        We truncate the fit if e.g. there is only a single frequency value to scan over
        but the fit is nominally set to order four. We additionally eliminate the
        constant term that should always be either zero or one.
        """
        cutoff_omega_d = min(len(self.omega_d_linspace), self.options.fit_cutoff)
        cutoff_amp = min(len(self.amp_linspace), self.options.fit_cutoff)
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

    def _polynomial_fit(
        self,
        xydata: np.ndarray,
        *fit_params: np.ndarray | tuple,
        bare_same: bool = False,
    ) -> np.ndarray | float:
        """Fit function to pass to curve fit, assume a 2D polynomial."""
        exp_pair_map = self.exponent_pair_idx_map
        omega_d, amp = xydata.T
        result = 1.0 if bare_same else 0.0
        for idx in exp_pair_map:
            exp_pair = exp_pair_map[idx]
            result += fit_params[idx] * omega_d ** exp_pair[0] * amp ** exp_pair[1]
        return result

    def displaced_state(
        self, omega_d: float, amp: float, coeffs: np.ndarray, state_idx: int
    ) -> qt.Qobj:
        """Construct the ideal displaced state, not including nonlinear transitions."""
        return sum(
            self._polynomial_fit(
                np.array([omega_d, amp]),
                *coeffs[state_idx_component, :],
                bare_same=state_idx == state_idx_component,
            )
            * qt.basis(self.num_states, state_idx_component)
            for state_idx_component in range(self.num_states)
        ).unit()

    def amp_to_range_idx(self, amp: float, omega_d: float) -> int:
        amp_idx = self.amp_to_idx(amp, omega_d)
        return int(np.floor(amp_idx / self._num_amp_pts_per_range))

    def omega_d_to_idx(self, omega_d: float) -> np.ndarray[int]:
        return np.argmin(np.abs(self.omega_d_linspace - omega_d))

    def amp_to_idx(self, amp: float, omega_d: float) -> np.ndarray[int]:
        omega_d_idx = self.omega_d_to_idx(omega_d)
        return np.argmin(np.abs(self.amp_linspace[:, omega_d_idx] - amp))
