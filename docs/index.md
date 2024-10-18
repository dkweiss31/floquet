# Getting started

**floquet**  is a python package for performing floquet simulations on quantum systems to identify nonlinear resonances.

## Installation

For now we support only installing directly from github
```bash
pip install git+https://github.com/dkweiss31/floquet
```

Requires Python 3.10+

## Example

Before jumping straight into the Floquet analysis, we first need to define our system Hamiltonian and the drive parameters. Here we take the example of a transmon and utilize the scubits library to help define the system Hamiltonian. Note however that the code accepts QuTiP `Qobj` as input for the Hamiltonian. 
```python
num_states = 20
qubit_params = {"EJ": 20.0, "EC": 0.2, "ng": 0.25, "ncut": 41}
tmon = scq.Transmon(**qubit_params, truncated_dim=num_states)
hilbert_space = scq.HilbertSpace([tmon])
hilbert_space.generate_lookup()
evals = hilbert_space["evals"][0][0:num_states]
# Define the Hamiltonian of the transmon in its eigenbasis, in which H0 is diagonal
H0 = 2.0 * np.pi * qt.Qobj(np.diag(evals - evals[0]))
# Use this helper method to define the charge-number operator in the dressed eigenbasis 
H1 = hilbert_space.op_in_dressed_eigenbasis(tmon.n_operator)
```
We then also need to specify the drive frequencies we plan to drive at, which correspond to the cavity frequencies
```python
omega_d_values = 2.0 * np.pi * np.linspace(7.5, 10.0, 120)
```
Finally, we specify the induced ac-stark shift that we want the qubit to experience. The dispersive coupling of a qubit to the cavity differs as a function of the cavity frequency. Thus we provide helper functions for normalizing the drive amplitudes so that we scan over the same induced ac-Stark shifts for all given frequencies. 
```python
import floquet as ft

state_indices = [0, 1]  # get data for ground and first excited states
chi_ac_linspace = 2.0 * np.pi * np.linspace(0.0, 0.1, 59) # 100 MHz ac-Stark shift
chi_to_amp = ft.ChiacToAmp(H0, H1, state_indices, omega_d_values)
drive_amplitudes = chi_to_amp.amplitudes_for_omega_d(chi_ac_linspace)
```
We can now pass these derived quantities to `Model` which specifies the model system
```python
model = ft.Model(
    H0, H1, omega_d_values=omega_d_values, drive_amplitudes=drive_amplitudes
)
```
We are now ready to create an instance of the `FloquetAnalysis` class, and run the full Floquet simulation
```python
options = ft.Options(num_cpus=6)
floquet_analysis = ft.FloquetAnalysis(model, state_indices=state_indices, options=options)
data_vals = floquet_analysis.run()
```
`data_vals` is a dictionary containing all quantities computed during the call to `run()`. This includes the overlap with the "ideal" displaced state, which can be plotted to reveal "scars" in the drive frequency and amplitude space where resonances occur. This part of the analysis is based on [Xiao, Venkatraman et al, arXiv (2023)](https://arxiv.org/abs/2304.13656), see Appendices I and J. Additionally we perform a so-called branch analysis to understand which states are responsible for ionization, based on [Dumas et al, arXiv 2024](https://arxiv.org/abs/2402.06615). See the tutorial notebooks under Examples on the left for example applications of the analysis, how to plot and visualize the computed quantities, etc.

## Citation

If you found this package useful in academic work, please cite

```bibtex
@misc{floquet2024,
  title  = {Floquet: Identifying nonlinear resonances in quantum systems due to parametric drives},
  author = {Daniel K. Weiss},
  year   = {2024},
  howpublished    = {\url{https://github.com/dkweiss31/floquet}}
}
```

Also please consider starring the project on [github](https://github.com/dkweiss31/floquet/)!