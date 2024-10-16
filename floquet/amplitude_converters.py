import numpy as np
import qutip as qt


class ChiacToAmp:
    r"""Convert given induced ac-stark shift values to drive amplitudes.

    Consider a qubit coupled to an oscillator with the interaction Hamiltonian
    $H_I = g(a + a^{\dagger})(b + b^{\dagger})$. If the oscillator is driven to
    an average occupation number of $\bar{n}$, then the effective drive strength
    seen by the qubit is $\Omega_d = 2 g \sqrt{\bar{n}}$. On the other hand based
    on a Schrieffer-Wolff transformation, the interaction hamiltonian is
    $H^{(2)} = \chi a^{\dagger}ab^{\dagger}b$. The average induced
    ac-stark shift is then $\chi_{ac} = \chi \bar{n}$. Thus $\Omega_d = 2g\sqrt{\chi_{\rm ac}/\chi}$.
    Observe that since $\chi \sim g^2$, $g$ effectively cancels out and can be set to 1.
    """  # noqa E501

    def __init__(
        self, H0: qt.Qobj, H1: qt.Qobj, state_indices: list, omega_d_values: np.ndarray
    ):
        self.H0 = H0
        self.H1 = H1
        self.state_indices = state_indices
        self.omega_d_linspace = omega_d_values

    def amplitudes_for_omega_d(self, chi_ac_linspace: np.ndarray) -> np.ndarray:
        r"""Return drive amplitudes corresponding to $\chi_{\rm ac}$ values."""
        if isinstance(chi_ac_linspace, float):
            chi_ac_linspace = np.array([chi_ac_linspace])
        chis_for_omega_d = self.compute_chis_for_omega_d()
        # Note that the below has the right units because chi_ac_linspace has units of
        # 2 pi GHz, while H1 is unitless. Thus chis_for_omega_d has units of
        # 1/(2 pi GHz) so the below has units of 2 pi GHz as required.
        return np.einsum(
            "a,w->aw", 2.0 * np.sqrt(chi_ac_linspace), 1.0 / np.sqrt(chis_for_omega_d)
        )

    def compute_chis_for_omega_d(self) -> np.ndarray:
        """Compute chi difference for the first two states in state_indices.

        Based on the analysis in Zhu et al PRB (2013)
        """
        chi_0 = self.chi_ell(
            np.diag(self.H0.full()),
            self.H1.full(),
            self.omega_d_linspace,
            self.state_indices[0],
        )
        chi_1 = self.chi_ell(
            np.diag(self.H0.full()),
            self.H1.full(),
            self.omega_d_linspace,
            self.state_indices[1],
        )
        return np.abs(chi_1 - chi_0)

    @staticmethod
    def chi_ell_ellp(
        energies: np.ndarray, H1: np.ndarray, E_osc: float, ell: int, ellp: int
    ) -> np.ndarray:
        E_ell_ellp = energies[ell] - energies[ellp]
        return np.abs(H1[ell, ellp]) ** 2 / (E_ell_ellp - E_osc)

    def chi_ell(
        self, energies: np.ndarray, H1: np.ndarray, E_osc: float, ell: int
    ) -> np.ndarray:
        n = len(energies)
        return sum(
            self.chi_ell_ellp(energies, H1, E_osc, ell, ellp)
            - self.chi_ell_ellp(energies, H1, E_osc, ellp, ell)
            for ellp in range(n)
        )


class XiSqToAmp:
    r"""Convert given $|\xi|^2$ value into a drive amplitude.

    This is based on the equivalence $\xi = 2 \Omega_d \omega_d / (\omega_d^2-\omega^2)$,
    where in this definition $|\xi|^2= 2 \chi_{\rm ac} / \alpha$ where $\chi_{\rm ac}$ is
    the induced ac stark shift, $\alpha$ is the anharmonicity and $\Omega_d$ is the
    drive amplitude.
    """  # noqa E501

    def __init__(
        self,
        H0: qt.Qobj,
        H1: qt.Qobj,
        state_indices: list,
        omega_d_linspace: np.ndarray,
    ):
        self.H0 = H0
        self.H1 = H1
        self.state_indices = state_indices
        self.omega_d_linspace = omega_d_linspace

    def amplitudes_for_omega_d(self, xi_sq_linspace: np.ndarray) -> np.ndarray:
        r"""Return drive amplitudes corresponding to $|\xi|^2$ values."""
        if isinstance(xi_sq_linspace, float):
            xi_sq_linspace = np.array([xi_sq_linspace])
        idx_0 = self.state_indices[0]
        idx_1 = self.state_indices[1]
        drive_matelem = self.H1[idx_0, idx_1]
        omega_01 = self.H0[idx_1, idx_1] - self.H0[idx_0, idx_0]
        return np.einsum(
            "x,w->xw",
            np.sqrt(xi_sq_linspace) / np.abs(drive_matelem),
            np.abs(omega_01**2 - self.omega_d_linspace**2)
            / (2 * self.omega_d_linspace),
        )
