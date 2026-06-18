import numpy as np
import qutip as qt
import scqubits as scq
import floquet as ft


def main():
    num_states = 20
    qubit_params = {"EJ": 20.0, "EC": 0.2, "ng": 0.25, "ncut": 41}
    tmon = scq.Transmon(**qubit_params, truncated_dim=num_states)
    state_indices = [0, 1]


    def get_H0_H1(qubit_instance: scq.GenericQubit) -> tuple[qt.Qobj, qt.Qobj]:
        hilbert_space = scq.HilbertSpace([qubit_instance])
        hilbert_space.generate_lookup()
        evals = hilbert_space["evals"][0][0:num_states]
        H0 = 2.0 * np.pi * qt.Qobj(np.diag(evals - evals[0]))
        H1 = hilbert_space.op_in_dressed_eigenbasis(qubit_instance.n_operator)
        return H0, H1


    H0, H1 = get_H0_H1(tmon)
    omega_d_values = 2.0 * np.pi * np.linspace(7.5, 10.0, 100)
    chi_ac_values = 2.0 * np.pi * np.linspace(0.0, 0.1, 100)
    chi_to_amp = ft.ChiacToAmp(H0, H1, state_indices, omega_d_values)
    drive_amplitudes = chi_to_amp.amplitudes_for_omega_d(chi_ac_values)

    model = ft.Model(
        H0, H1, omega_d_values=omega_d_values, drive_amplitudes=drive_amplitudes
    )

    options = ft.Options(
        fit_range_fraction=0.5,
        floquet_sampling_time_fraction=0.0,
        fit_cutoff=4,
        overlap_cutoff=0.8,
        nsteps=30_000,
        num_cpus=4,
        save_floquet_modes=True,
    )

    floquet_analysis = ft.FloquetAnalysis(
        model, state_indices=state_indices, options=options
    )
    data_vals = floquet_analysis.run()
if __name__ == "__main__":
    main()