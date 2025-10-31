from __future__ import annotations

import importlib
import inspect
import pathlib
import re

import h5py
import numpy as np
import qutip as qt


class Serializable:
    """Mixin class for reading and writing to file using h5py."""

    def __new__(cls: Serializable, *args, **kwargs) -> Serializable:  # noqa ARG003
        """Records which parameters should be saved so they can be passed to init."""
        cls._init_attrs = get_init_params(cls)
        return super().__new__(cls)

    def __eq__(self, other: Serializable) -> bool:
        init_params = get_init_params(self)
        equal = True
        for key in init_params:
            self_val = getattr(self, key)
            other_val = getattr(other, key)
            if isinstance(self_val, np.ndarray):
                equal = equal and np.allclose(self_val, other_val)
            else:
                equal = equal and self_val == other_val
        return equal

    def __hash__(self) -> int:
        init_params = get_init_params(self)
        vals = ()
        for key in init_params:
            _val = getattr(self, key)
            val = tuple(_val) if isinstance(_val, np.ndarray) else _val
            vals = (*vals, val)
        return hash(vals)

    def __str__(self) -> str:
        return "\n".join(
            f"{k}: {getattr(self, k)}"
            for k in self._init_attrs
            if getattr(self, k) is not None
        )

    def serialize(self) -> dict:
        """Serialize a class so that it is ready to be written.

        This method creates nested dictionaries appropriate for writing to h5 files.
        Importantly, we save metadata associated with the class itself and any classes
        it takes as input so that they can be reinstantiated later.

        Returns:
            initdata: Dictionary of data to save, in a format appropriate to pass to h5
        """
        initdata = {name: getattr(self, name) for name in self._init_attrs}
        for _key, _val in initdata.items():
            if hasattr(_val, "_init_attrs"):
                # this is itself a class that defines _init_attrs
                initdata[_key] = _val.serialize() | {
                    "name_and_module": (
                        _val.__class__.__name__,
                        _val.__class__.__module__,
                    )
                }
            elif isinstance(_val, qt.Qobj):
                initdata[_key] = _val.data.to_array()
            else:
                initdata[_key] = _val
        return initdata

    def write_to_file(
        self, filepath: str | pathlib.Path, data_dict: dict | None = None
    ):
        """Write a class and associated data to file.

        The goal is to be able to read back both the data that was saved and all of the
        data necessary to reinitialize the class.

        Parameters:
            filepath: Path to the file where we want to save the data. Must be an h5 or
                h5py file
            data_dict: Dictionary containing various raw data to save
        """
        if data_dict is None:
            data_dict = {}
        with h5py.File(filepath, "w") as f:
            for key, val in data_dict.items():
                f.create_dataset(key, data=val, track_order=True)

            def _write(_key: str, _val: h5py.Dataset, _group: h5py.Group) -> None:
                if isinstance(_val, dict):
                    # Could be a real dictionary, or a class in disguise
                    _key_group = _group.create_group(_key, track_order=True)
                    name_and_module = _val.pop("name_and_module", None)
                    if name_and_module is not None:
                        # Gasp! Its a class!
                        _key_group.attrs["name_and_module"] = name_and_module
                    for __key, __val in _val.items():
                        _write(__key, __val, _key_group)
                elif _val is not None:
                    _group.create_dataset(_key, data=_val, track_order=True)
                else:
                    pass

            initdata = self.serialize()
            param_grp = f.create_group("_init_attrs", track_order=True)
            # Save metadata of parent class
            param_grp.attrs["name_and_module"] = (
                self.__class__.__name__,
                self.__class__.__module__,
            )
            for key, val in initdata.items():
                _write(key, val, param_grp)


def read_from_file(filepath: str | pathlib.Path) -> tuple[Serializable, dict]:
    """Read a class and associated data from file.

    Parameters:
        filepath: Path to the file containing both raw data and the information needed
            to reinitialize our class

    Returns:
        new_class_instance: Class that inherits from Serializable that was earlier
            written with its method write_to_file
        data_dict: Dictionary of data that was passed to write_to_file at the time
    """
    data_dict = {}
    with h5py.File(filepath, "r") as f:
        # read data that doesn't have to do with class reinitialization
        for key in f:
            if key != "_init_attrs":
                data_dict[key] = f[key][()]

        def _read(_key: str, _group: h5py.Group):  # noqa ANN202
            # indicates that _group is a class that needs to be instantiated
            if "name_and_module" in _group[_key].attrs:
                _name, _module = _group[_key].attrs["name_and_module"]
                module = importlib.import_module(_module)
                class_instance = getattr(module, _name)
                # recurse through in case some classes also take other classes as input
                _init_dict = {}
                for __key in _group[_key]:
                    _init_dict[__key] = _read(__key, _group[_key])
                return class_instance(**_init_dict)
            if isinstance(_group[_key], h5py.Group):
                # This means it really is a dictionary, and not a class that wants to be
                # reinitialized
                _init_dict = {}
                for __key in _group[_key]:
                    _init_dict[__key] = _read(__key, _group[_key])
                return _init_dict
            return _group[_key][()]

        if "_init_attrs" in f:
            new_class_instance = _read("_init_attrs", f)
        else:
            raise ValueError(
                f"file {filepath} does not have an attribute '_init_attrs', "
                f"indicating it was not saved with the method write_to_file"
                f" of the class you are trying to reinstantiate."
            )

    return new_class_instance, data_dict


def get_init_params(obj: Serializable) -> list[str]:
    """Returns a list of parameters entering `__init__` of `obj`."""
    init_params = list(inspect.signature(obj.__init__).parameters.keys())
    if "self" in init_params:
        init_params.remove("self")
    if "kwargs" in init_params:
        init_params.remove("kwargs")
    return init_params


def generate_file_path(extension: str, file_name: str, path: str) -> str:
    # Ensure the path exists.
    path_obj = pathlib.Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for file_path in path_obj.glob("*"):
        if f"_{file_name}.{extension}" in file_path.name:
            match = re.match(r"(\d+)_", file_path.stem)
            if match:
                numeric_prefix = int(match.group(1))
                max_numeric_prefix = max(numeric_prefix, max_numeric_prefix)

    new_file_name = f"{str(max_numeric_prefix + 1).zfill(5)}_{file_name}.{extension}"
    return str(path_obj / new_file_name)
