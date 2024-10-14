# Python API

The **floquet** Python API consists largely of the **floquet_analysis** function which initializes a **FloquetAnalysis** class. We can then call `run(filepath)` on instances of this class to perform the Floquet simulation. The computed data is then saved both as attributes of the class and also to the file specified by `filepath`, which should be a string specifying a `h5py` file.

## Floquet methods

::: floquet.floquet
    options:
        show_source: false

## Displaced state

::: floquet.displaced_state
    options:
        show_source: false

## Model

::: floquet.model
    options:
        show_source: false

## Amplitude conversion utilities

::: floquet.amplitude_converters
    options:
        show_source: false

## Options

::: floquet.options
    options:
        show_source: false

## File utilities

::: floquet.utils.file_io
    options:
        show_source: false
