from typing import Any

import numpy as np

from raygent.savers import Saver

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class HDF5Saver(Saver):
    """
    A saver that writes data to an HDF5 file.

    Example usage:
        saver = HDF5Saver("my_data.h5", dataset_name="my_dataset")
    """

    def __init__(
        self, file_path: str, dataset_name: str = "dataset", approach: str = "append"
    ):
        """
        Args:
            file_path: The path to the HDF5 file.
            dataset_name: Name of the dataset within the HDF5 file.
            approach: One of `'append'`, `'overwrite'`, or `'update'`.
        """
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.approach = approach.strip().lower()

    def save(
        self, data: list[Any], indices: Any | None = None, **kwargs: dict[str, Any]
    ) -> None:
        """
        Saves the data to an HDF5 dataset. Appends if the dataset already exists.

        Args:
            data: A list of results to save.
            **kwargs: Additional parameters for the HDF5 saving process.
        """
        if not HAS_H5PY:
            raise ImportError("H5PY is not installed.")

        arr = np.array(data, dtype=kwargs.get("dtype"))

        with h5py.File(self.file_path, "a") as h5file:
            if self.approach == "append":
                if self.dataset_name not in h5file:
                    dset = h5file.create_dataset(
                        self.dataset_name, data=arr, maxshape=(None,)
                    )
                else:
                    dset = h5file[self.dataset_name]
                    old_size = dset.shape[0]
                    new_size = old_size + arr.shape[0]
                    dset.resize(new_size, axis=0)
                    dset[old_size:new_size] = arr

            elif self.approach == "overwrite":
                if self.dataset_name in h5file:
                    del h5file[self.dataset_name]
                h5file.create_dataset(self.dataset_name, data=arr)

            elif self.approach == "update":
                if self.dataset_name not in h5file:
                    raise ValueError(
                        "Cannot update because the dataset does not exist."
                    )
                dset = h5file[self.dataset_name]
                dset[indices] = arr

            else:
                raise ValueError(f"Unknown approach '{self.approach}'.")
