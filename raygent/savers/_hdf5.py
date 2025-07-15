from typing import Any

import numpy as np

from raygent.savers import Saver

try:
    import h5py

    has_h5py = True
except ImportError:
    has_h5py = False


class HDF5Saver(Saver):
    """A saver that writes data to an HDF5 file.

    `HDF5Saver` provides functionality to persist computational results in the HDF5
    format, which is designed for storing and managing large, complex data collections.
    HDF5 offers a hierarchical structure (similar to a file system) and supports
    efficient I/O operations for both small and large datasets.

    This implementation supports three approaches to saving data:

    - `append`: Add new data to the existing dataset (creates if not exists)
    - `overwrite`: Replace existing dataset with new data
    - `update`: Update specific indices in the existing dataset with new values

    HDF5 format offers several advantages:

    - Efficient storage of large, heterogeneous datasets
    - Partial I/O operations (reading/writing subsets of data)
    - Built-in compression
    - Self-describing format with rich metadata support
    - Cross-platform compatibility

    Requirements:
        This saver requires the h5py package to be installed:

        ```sh
        pip install h5py
        ```

    Examples:
        Basic usage:

        ```python
        # Create an HDF5Saver for storing results
        saver = HDF5Saver("results.h5", dataset_name="experiment_1")

        # Use with TaskManager
        task_manager = TaskManager(MyTask)
        task_manager.submit_tasks(batch, saver=saver, save_interval=100)
        ```

        Overwriting existing data:

        ```python
        # Create a saver that overwrites existing data
        saver = HDF5Saver(
            "daily_metrics.h5", dataset_name="day_20240306", approach="overwrite"
        )

        # Save new results, replacing any existing dataset
        results = process_batch(today_data)
        saver.save(results)
        ```

        Updating specific indices:

        ```python
        # Create a saver for updating existing data
        saver = HDF5Saver(
            "time_series.h5", dataset_name="sensor_readings", approach="update"
        )

        # Update specific time indices with new values
        new_data = [99.5, 98.3, 97.8]
        indices = [10, 20, 30]  # Positions to update
        saver.save(new_data, indices=indices)
        ```

        Multiple datasets in one file:

        ```python
        # Create multiple savers with different dataset names
        saver1 = HDF5Saver("project_data.h5", dataset_name="raw_data")
        saver2 = HDF5Saver("project_data.h5", dataset_name="processed_data")
        saver3 = HDF5Saver("project_data.h5", dataset_name="metadata")

        # Save different types of data to the same file
        saver1.save(raw_measurements)
        saver2.save(processed_results)
        saver3.save(experiment_metadata)
        ```

    Notes:
        - HDF5 is well-suited for scientific computing and applications that need to
          store large numerical arrays efficiently.
        - Unlike .npy files, HDF5 allows for efficient partial I/O operations, making
          it suitable for datasets that are too large to fit in memory.
        - HDF5 files can store multiple datasets with different names, allowing
          related data to be kept in a single file.
        - HDF5 has limitations with parallel writes from multiple processes. For
          highly parallel workloads, consider ZarrSaver instead.
    """

    def __init__(
        self, file_path: str, dataset_name: str = "dataset", approach: str = "append"
    ):
        """Initialize an HDF5Saver instance.

        Args:
            file_path: The path to the HDF5 file where data will be saved.
            dataset_name: Name of the dataset within the HDF5 file. This allows
                multiple datasets to be stored in a single HDF5 file.
                Default is "dataset".
            approach: One of 'append', 'overwrite', or 'update', determining how data is
                saved when the dataset already exists. Default is "append".

        Notes:
            - The file_path should have the .h5 or .hdf5 extension by convention.
            - Dataset names can use a path-like syntax with forward slashes to create
              groups (similar to directories) within the HDF5 file, e.g.,
              "experiments/trial_1/data".
        """
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.approach = approach.strip().lower()

    def save(
        self, data: list[Any], indices: Any | None = None, **kwargs: dict[str, Any]
    ) -> None:
        """Saves the data to an HDF5 dataset according to the specified approach.

        This method implements the abstract save method from the Saver base class.
        It persists the provided data to an HDF5 dataset using the configured approach
        (`append`, `overwrite`, or `update`).

        The method handles creating new datasets, extending existing datasets, or
        updating specific indices in existing datasets. It automatically converts
        the input data to a numpy array before saving.

        Args:
            data: A list of results to save. The data will be converted to a numpy
                array before saving.
            indices: Required when approach is 'update', specifies the indices where
                data should be written in the existing dataset. Must be compatible with
                the shape of the input data.
            **kwargs: Additional keyword arguments for configuring the save operation.
                Common options include:
                - dtype: Data type for the numpy array conversion
                - compression: Compression filter to use (e.g., "gzip", "lzf")
                - compression_opts: Options for the compression filter
                - batches: Batch shape for the dataset

        Raises:
            ImportError: If the h5py library is not installed.
            ValueError: If approach is 'update' but the dataset does not exist.
            ValueError: If an unknown approach is specified.
            TypeError: If the data cannot be converted to a numpy array.

        Examples:
            Saving data with the append approach:

            ```python
            saver = HDF5Saver("results.h5", dataset_name="measurements")

            # First save creates the dataset
            saver.save([1, 2, 3, 4, 5])

            # Subsequent saves append to it
            saver.save([6, 7, 8, 9, 10])
            ```

            Saving with compression:

            ```python
            saver = HDF5Saver("compressed_results.h5", dataset_name="large_dataset")

            # Save with gzip compression
            saver.save(
                large_data_array,
                dtype="float32",
                compression="gzip",
                compression_opts=9,
                batches=(100,),
            )
            ```

            Updating specific indices:

            ```python
            saver = HDF5Saver(
                "values.h5", dataset_name="sensor_data", approach="update"
            )

            # Update specific positions in an existing dataset
            new_values = [99.5, 98.3, 97.8]
            indices = slice(10, 13)  # Update positions 10, 11, 12
            saver.save(new_values, indices=indices)
            ```

        Notes:
            - The append operation efficiently extends the dataset without loading
              the entire existing data into memory, making it suitable for large datasets.
            - When using compression, consider the tradeoff between storage space
              and read/write performance.
            - HDF5 supports resizable datasets with the maxshape parameter, which
              is automatically configured for append operations.
            - For optimal performance with large datasets, configure appropriate
              batch sizes based on expected access patterns.
        """
        if not has_h5py:
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
