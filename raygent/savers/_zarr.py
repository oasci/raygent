from typing import Any, Literal
import numpy as np
from raygent.savers import Saver

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False


class ZarrSaver(Saver):
    """
    A saver that writes data to a Zarr array.

    Example usage:
        saver = ZarrSaver("my_data.zarr", dataset_name="my_dataset")
    """

    def __init__(
        self,
        file_path: str,
        dataset_name: str = "dataset",
        approach: Literal["append", "overwrite", "update"] = "append",
    ):
        """
        Args:
            file_path: The path to the Zarr container to create or open.
            dataset_name: Name of the dataset within the Zarr store.
            approach: One of `'append'`, `'overwrite'`, or `'update'`.
        """
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.approach = approach.strip().lower()

    def save(
        self, data: list[Any], indices: Any | None = None, **kwargs: dict[str, Any]
    ) -> None:
        """
        Saves the data to a Zarr array according to the specified approach.

        Args:
            data: A list of results to save.
            indices: Indices to write data if approach is `update`.
        """
        if not HAS_ZARR:
            raise ImportError("Zarr is not installed.")

        arr = np.array(data)

        store = zarr.storage.LocalStore(self.file_path, read_only=False)
        print(store)

        if self.approach == "append":
            try:
                z = zarr.open_array(store, path=self.dataset_name)
                z.append(arr)
            except FileNotFoundError:
                z = zarr.create_array(
                    store, name=self.dataset_name, data=arr, overwrite=True
                )
        elif self.approach == "overwrite":
            z = zarr.create_array(
                store, name=self.dataset_name, data=arr, overwrite=True
            )
        elif self.approach == "update":
            z = zarr.open_array(store, path=self.dataset_name)
            if not indices:
                raise ValueError("`indices` cannot be None with 'update'")
            z[indices] = arr

        else:
            raise ValueError(
                f"Unknown approach '{self.approach}'. Use 'append', 'overwrite', or 'update'."
            )
