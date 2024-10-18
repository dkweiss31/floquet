import itertools
import pathlib

import numpy as np
import pytest
import qutip as qt
import scqubits as scq

from floquet import (
    ChiacToAmp,
    DisplacedState,
    FloquetAnalysis,
    Model,
    Options,
    read_from_file,
    XiSqToAmp,
)


def _filepath(path: pathlib.Path) -> pathlib.Path:
    d = path / "sub"
    d.mkdir()
    return d / "tmp.h5py"


@pytest.fixture(scope="session", autouse=True)
def setup_floquet() -> tuple:
    num_states = 19
    EJ = 29.0
    EC = 0.155
    ng = 0.0
    ncut = 21
    INIT_DATA_TO_SAVE = {"EJ": EJ, "EC": EC, "ng": ng, "ncut": ncut}
    tmon = scq.Transmon(EJ=29.0, EC=0.155, ng=0.0, ncut=21, truncated_dim=num_states)
    omega_d_values = 2.0 * np.pi * np.linspace(6.9, 13, 39)
    chi_ac_linspace = 2.0 * np.pi * np.linspace(0.0, 0.2, 40)
    state_indices = [0, 1, 3]

    hilbert_space = scq.HilbertSpace([tmon])
    hilbert_space.generate_lookup()
    evals = hilbert_space["evals"][0][0:num_states]
    H0 = 2.0 * np.pi * qt.Qobj(np.diag(evals - evals[0]))
    H1 = hilbert_space.op_in_dressed_eigenbasis(tmon.n_operator)

    options = Options(fit_range_fraction=0.5, num_cpus=6)
    chi_to_amp = ChiacToAmp(H0, H1, state_indices, omega_d_values)
    drive_amplitudes = chi_to_amp.amplitudes_for_omega_d(chi_ac_linspace)
    model = Model(H0, H1, omega_d_values, drive_amplitudes)
    floquet_transmon = FloquetAnalysis(
        model, state_indices, options=options, init_data_to_save=INIT_DATA_TO_SAVE
    )
    return floquet_transmon, chi_to_amp, chi_ac_linspace


def test_chi_vs_xi(setup_floquet: tuple):
    r"""Test that $chi_{\rm ac}$ and $|\xi|^2$ calculations agree.

    They are related by the formula $\frac{1}{2}|\xi|^2=\chi_{\rm ac}/E_{C}$.
    """
    floquet_transmon, _, chi_ac_linspace = setup_floquet
    amps_from_chi_ac = floquet_transmon.model.drive_amplitudes
    EC = floquet_transmon.init_data_to_save["EC"]
    xi_sq_linspace = 2.0 * chi_ac_linspace / EC / 2 / np.pi
    xi_sq_to_amp = XiSqToAmp(
        floquet_transmon.model.H0,
        floquet_transmon.model.H1,
        floquet_transmon.state_indices,
        floquet_transmon.model.omega_d_values,
    )
    amps_from_xi_sq = xi_sq_to_amp.amplitudes_for_omega_d(xi_sq_linspace)
    rel_diff = np.abs(
        (amps_from_xi_sq[1:] - amps_from_chi_ac[1:]) / amps_from_xi_sq[1:]
    )
    assert np.max(rel_diff < 0.05)


def test_displaced_fit_and_reinit(setup_floquet: tuple, tmp_path: pathlib.Path):
    floquet_transmon, chi_to_amp, _ = setup_floquet
    filepath = _filepath(tmp_path)
    data_dict = floquet_transmon.run(filepath=filepath)
    # for these random pairs, overlap should be near unity (most
    # pairs don't correspond to a resonance!)
    omega_d_vals = 2.0 * np.pi * np.array([7.5, 9.0, 11.8, 12.7])
    chi_ac_vals = np.array([0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.19])
    omega_d_chi_ac = itertools.product(omega_d_vals, chi_ac_vals)
    for omega_d, chi_ac in omega_d_chi_ac:
        omega_d_idx = floquet_transmon.model.omega_d_to_idx(omega_d)
        amp = chi_to_amp.amplitudes_for_omega_d(chi_ac)[0, omega_d_idx]
        for array_idx, state_idx in enumerate(floquet_transmon.state_indices):
            disp_coeffs = data_dict["fit_data"]
            displaced_state = DisplacedState(
                floquet_transmon.hilbert_dim,
                floquet_transmon.model,
                floquet_transmon.state_indices,
                floquet_transmon.options,
            )
            disp_gs = displaced_state.displaced_state(
                omega_d, amp, state_idx=state_idx, coefficients=disp_coeffs[array_idx]
            )
            f_modes_energies = floquet_transmon.run_one_floquet((omega_d, amp))
            floquet_mode = floquet_transmon.identify_floquet_modes(
                f_modes_energies, (omega_d, amp), displaced_state, disp_coeffs
            )
            overlap = np.abs(qt.Qobj(floquet_mode[array_idx, 1:]).dag() * disp_gs)
            assert 0.98 < overlap < 1.0
    floquet_transmon.write_to_file(filepath, data_dict)
    new_floquet_transmon, new_data_dict = read_from_file(filepath)
    assert new_floquet_transmon == floquet_transmon
    for key in data_dict:
        assert np.allclose(data_dict[key], new_data_dict[key])


def test_one_D_amplitudes(setup_floquet: tuple):
    floquet_transmon, _, _ = setup_floquet
    n_omega_d = len(floquet_transmon.model.omega_d_values)
    n_amps = len(floquet_transmon.model.drive_amplitudes)
    drive_amplitudes = 2.0 * np.pi * np.linspace(0.0, 6.0, n_amps)
    model = Model(
        floquet_transmon.model.H0,
        floquet_transmon.model.H1,
        floquet_transmon.model.omega_d_values,
        drive_amplitudes,
    )
    assert model.drive_amplitudes.shape == (n_amps, n_omega_d)
    # Compare all drive amplitudes to those at a specific frequency, ensure they are
    # all identical
    equal_drive_for_omega_d = (
        model.drive_amplitudes == model.drive_amplitudes[:, 0, None]
    )
    assert equal_drive_for_omega_d.all()


def test_reinit(setup_floquet: tuple, tmp_path: pathlib.Path):
    floquet_transmon, _, _ = setup_floquet
    filepath = _filepath(tmp_path)
    floquet_transmon.write_to_file(filepath, {})
    new_floquet_transmon, _ = read_from_file(filepath)
    assert new_floquet_transmon == floquet_transmon


def test_displaced_bare_state(setup_floquet: tuple):
    floquet_transmon, _, _ = setup_floquet
    displaced_state = DisplacedState(
        floquet_transmon.hilbert_dim,
        floquet_transmon.model,
        floquet_transmon.state_indices,
        floquet_transmon.options,
    )
    for state_idx in floquet_transmon.state_indices:
        ideal_state = qt.basis(floquet_transmon.hilbert_dim, state_idx)
        disp_coeffs_bare = displaced_state.bare_state_coefficients(state_idx)
        # omega_d and amp values shouldn't matter
        calc_state = displaced_state.displaced_state(
            0.0, 0.0, state_idx, disp_coeffs_bare
        )
        assert calc_state == ideal_state
