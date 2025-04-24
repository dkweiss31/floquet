# FAQ

## How do I read my saved data?

If you have run a previous simulation where you saved your data to `"foo.h5py"`, by calling e.g.

>>> data_vals = floquet_analysis.run(filepath="foo.h5py") # doctest: +SKIP

the saved data and `FloquetAnalysis` instance that was used to produce this data can be read by calling

>>> read_floquet_instance, read_data = ft.read_from_file("foo.h5py") # doctest: +SKIP


## I'm seeing weird artifacts in the displaced state overlap plots

This issue occurs because the floquet mode has been displaced far from the origin, causing the overlap of the floquet mode with the bare state to fall below the threshold `overlap_cutoff`. This issue typically can be dealt with by decreasing `fit_range_fraction` from 1.0 to e.g. 0.5 or 0.25 (note that the fraction doesn't need to evenly divide 1.0). If 0.5 is chosen, the ideal-displaced-state fitting restarts after the first half of the drive amplitude values. The displaced state fitted from the first half range is then utilized to calculate overlaps for the states in the second half range. `fit_range_fraction` can be reduced arbitrarily, but beware of decreasing it too much or there won't be enough data to generate a faithful fit of the true displaced state.
