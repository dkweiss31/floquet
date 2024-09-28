import itertools

import numpy as np
import pytest
import qutip as qt
import scqubits as scq

from floquet import (
    ChiacToAmp,
    Options,
    XiSqToAmp,
    floquet_analysis,
    floquet_analysis_from_file,
    DriveParameters
)


def _filepath(path):
    d = path / 'sub'
    d.mkdir()
    return d / 'tmp.h5py'


@pytest.fixture(scope='session', autouse=True)
def setup_floquet():
    num_states = 20
    EJ = 29.0
    EC = 0.155
    ng = 0.0
    ncut = 21
    INIT_DATA_TO_SAVE = {'EJ': EJ, 'EC': EC, 'ng': ng, 'ncut': ncut}
    tmon = scq.Transmon(EJ=29.0, EC=0.155, ng=0.0, ncut=21, truncated_dim=num_states)
    omega_d_values = 2.0 * np.pi * np.linspace(6.9, 13, 39)
    chi_ac_linspace = 2.0 * np.pi * np.linspace(0.0, 0.2, 40)
    state_indices = [0, 1, 3]

    hilbert_space = scq.HilbertSpace([tmon])
    hilbert_space.generate_lookup()
    evals = hilbert_space['evals'][0][0:num_states]
    H0 = 2.0 * np.pi * qt.Qobj(np.diag(evals - evals[0]))
    H1 = hilbert_space.op_in_dressed_eigenbasis(tmon.n_operator)

    options = Options(fit_range_fraction=0.5, num_cpus=6)
    chi_to_amp = ChiacToAmp(H0, H1, state_indices, omega_d_values)
    drive_amplitudes = chi_to_amp.amplitudes_for_omega_d(chi_ac_linspace)
    drive_parameters = DriveParameters(omega_d_values, drive_amplitudes)
    floquet_transmon = floquet_analysis(
        H0,
        H1,
        drive_parameters,
        state_indices=state_indices,
        options=options,
        init_data_to_save=INIT_DATA_TO_SAVE,
    )
    return floquet_transmon, chi_to_amp, chi_ac_linspace


def test_chi_vs_xi(setup_floquet):
    floquet_transmon, _, chi_ac_linspace = setup_floquet
    amps_from_chi_ac = floquet_transmon.drive_parameters.drive_amplitudes
    EC = floquet_transmon.init_data_to_save['EC']
    xi_sq_linspace = 2.0 * chi_ac_linspace / EC / 2 / np.pi
    xi_sq_to_amp = XiSqToAmp(
        floquet_transmon.H0,
        floquet_transmon.H1,
        floquet_transmon.state_indices,
        floquet_transmon.drive_parameters.omega_d_values,
    )
    amps_from_xi_sq = xi_sq_to_amp.amplitudes_for_omega_d(xi_sq_linspace)
    rel_diff = np.abs(
        (amps_from_xi_sq[1:] - amps_from_chi_ac[1:]) / amps_from_xi_sq[1:]
    )
    assert np.max(rel_diff < 0.05)


def test_displaced_fit_and_reinit(setup_floquet, tmp_path):
    floquet_transmon, chi_to_amp, _ = setup_floquet
    filepath = _filepath(tmp_path)
    floquet_transmon.run(filepath=filepath)
    # for these random pairs, overlap should be near unity (most
    # pairs don't correspond to a resonance!)
    omega_d_vals = 2.0 * np.pi * np.array([7.5, 9.0, 11.8, 12.7])
    chi_ac_vals = np.array([0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.19])
    omega_d_chi_ac = itertools.product(omega_d_vals, chi_ac_vals)
    for omega_d, chi_ac in omega_d_chi_ac:
        omega_d_idx = floquet_transmon.drive_parameters.omega_d_to_idx(omega_d)
        amp = chi_to_amp.amplitudes_for_omega_d(chi_ac)[0, omega_d_idx]
        for arr_idx, state_idx in enumerate(floquet_transmon.state_indices):
            disp_coeffs = floquet_transmon.coefficient_matrix
            disp_gs = floquet_transmon.displaced_state(
                omega_d, amp, disp_coeffs[arr_idx], state_idx=state_idx
            )
            f_modes_energies = floquet_transmon.run_one_floquet((omega_d, amp))
            floquet_mode = floquet_transmon.calculate_modes_quasies_ovlps(
                f_modes_energies, (omega_d, amp), disp_coeffs
            )
            overlap = np.abs(
                (qt.Qobj(floquet_mode[arr_idx, 3:]).dag() * disp_gs).data.toarray()[
                    0, 0
                ]
            )
            assert 0.98 < overlap < 1.0
    # reinit
    reinit_floquet_transmon = floquet_analysis_from_file(filepath)
    assert reinit_floquet_transmon.get_initdata() == floquet_transmon.get_initdata()


def test_displaced_bare_state(setup_floquet):
    floquet_transmon, chi_to_amp, _ = setup_floquet
    for state_idx in floquet_transmon.state_indices:
        ideal_state = qt.basis(floquet_transmon.num_states, state_idx)
        disp_coeffs_bare = floquet_transmon.bare_state_coefficients(state_idx)
        # omega_d and amp values shouldn't matter
        calc_state = floquet_transmon.displaced_state(
            0.0, 0.0, disp_coeffs_bare, state_idx=state_idx
        )
        assert calc_state == ideal_state
