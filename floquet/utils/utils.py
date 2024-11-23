import numpy as np


def create_exponent_pair_idx_map(cutoff_omega_d, cutoff_amp, fit_cutoff) -> dict:
    """Create dictionary of terms in polynomial that we fit.

    We truncate the fit if e.g. there is only a single frequency value to scan over
    but the fit is nominally set to order four. We additionally eliminate the
    constant term that should always be either zero or one.
    """
    idx_exp_map = [
        (idx_1, idx_2) for idx_1 in range(cutoff_omega_d) for idx_2 in range(cutoff_amp)
    ]
    # Kill constant term, which should always be 1 or 0 depending
    # on if the component is the same as the state being fit.
    # Moreover kill any terms depending only on drive frequency, since these
    # coefficients must be zero as the states have to agree at zero drive strength.
    for idx in range(cutoff_omega_d):
        idx_exp_map.remove((idx, 0))
    weighted_vals = [1.01 * idx_1 + idx_2 for (idx_1, idx_2) in idx_exp_map]
    sorted_idxs = np.argsort(weighted_vals)
    sorted_idx_exp_map = {}
    counter = 0
    for sorted_idx in sorted_idxs:
        exponents = idx_exp_map[sorted_idx]
        if sum(exponents) <= fit_cutoff:
            sorted_idx_exp_map[counter] = exponents
            counter += 1
    return sorted_idx_exp_map
