from __future__ import annotations

from .utils.file_io import Serializable


class Options(Serializable):
    def __init__(
        self,
        fit_range_fraction: float = 1.0,
        floquet_sampling_time_fraction: float = 0.0,
        fit_cutoff: int = 4,
        overlap_cutoff: float = 0.8,
        nsteps: int = 30_000,
        num_cpus: int = 1,
        save_floquet_modes: bool = False,
    ):
        if fit_range_fraction <= 0 or fit_range_fraction > 1:
            raise ValueError(
                f"Must have 0 < fit_range_fraction <= 1 but got {fit_range_fraction}"
            )
        self.fit_range_fraction = fit_range_fraction
        self.floquet_sampling_time_fraction = floquet_sampling_time_fraction
        if fit_cutoff < 1:
            raise ValueError(f"Must have fit_cutoff > 0 but got {fit_cutoff}")
        self.fit_cutoff = fit_cutoff
        if overlap_cutoff > 1 or overlap_cutoff < 0.7:
            raise ValueError(
                f"Must have 0.7 <= overlap_cutoff <= 1 but got {overlap_cutoff}"
            )
        self.overlap_cutoff = overlap_cutoff
        self.nsteps = nsteps
        self.num_cpus = num_cpus
        self.save_floquet_modes = save_floquet_modes
