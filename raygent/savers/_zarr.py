# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scienting Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.


from typing import Any, Literal

import numpy as np

from raygent.savers import Saver

try:
    import zarr

    has_zarr = True
except ImportError:
    has_zarr = False


class ZarrSaver(Saver):
    """A saver that writes data to a Zarr array.

    `ZarrSaver` provides functionality to persist computational results in
    [Zarr](https://zarr.dev/) format, which offers efficient batched,
    compressed, N-dimensional array storage. [Zarr](https://zarr.dev/) is
    particularly well-suited for large arrays that don't fit in memory and
    for cloud-based storage, supporting both local and remote persistence.

    This implementation supports three approaches to saving data:

    - append: Add new data to the existing array (creates if not exists).
    - overwrite: Replace existing data with new data.
    - update: Update specific indices in the existing array with new values.

    Zarr offers advantages over other formats for specific use cases:

    - Parallel read/write access for distributed computing.
    - Efficient access to subsets of large arrays.
    - Support for cloud storage backends (S3, GCS, etc.).
    - Good compression options for numerical data.

    Requirements:
        This saver requires the zarr package to be installed:

        ```sh
        pip install zarr
        ```

    Examples:
        Basic usage with local storage:

        ```python
        # Create a ZarrSaver for appending data
        saver = ZarrSaver("results.zarr", dataset_name="experiment_1")

        # Use with TaskRunner
        task_runner = TaskRunner(MyTask, ResultsCollector)
        task_runner.submit_tasks(batch, saver=saver, save_interval=100)
        ```

        Overwriting existing data:

        ```python
        # Create a saver that overwrites existing data
        saver = ZarrSaver(
            "results.zarr", dataset_name="daily_metrics", approach="overwrite"
        )

        # Save a batch of results directly
        results = process_batch(today_data)
        saver.save(results)
        ```

        Updating specific indices:

        ```python
        # Create a saver for updating existing data
        saver = ZarrSaver("results.zarr", dataset_name="time_series", approach="update")

        # Update specific time indices with new values
        new_data = calculate_corrections(raw_data)
        indices = [5, 10, 15, 20]  # Indices to update
        saver.save(new_data, indices=indices)
        ```

        Using with a cloud storage backend (requires appropriate zarr plugins):

        ```python
        # Using with AWS S3 (requires s3fs)
        import s3fs

        s3 = s3fs.S3FileSystem(anon=False)
        store = s3fs.S3Map(root="mybucket/zarr-data", s3=s3, check=False)

        # Create a ZarrSaver with the S3 location
        saver = ZarrSaver(store, dataset_name="remote_dataset", approach="append")
        ```

    Notes:
        -   Zarr is particularly well-suited for large-scale numerical data and
            distributed computing workloads.
        -   For optimal performance, consider batch size carefully based on access
            patterns.
        -   Unlike HDF5, Zarr allows concurrent reads and writes from multiple processes
            or machines, making it ideal for distributed computing.
        -   The `update` approach requires that the dataset already exists and that
            valid indices are provided.
    """

    def __init__(
        self,
        file_path: str,
        dataset_name: str = "dataset",
        approach: Literal["append", "overwrite", "update"] = "append",
    ):
        """Initialize a ZarrSaver instance.

        Args:
            file_path: The path to the Zarr container to create or open. This can be
                a local path or a URL to a supported remote storage backend.
            dataset_name: Name of the dataset within the Zarr store.
            approach: One of `'append'`, `'overwrite'`, or `'update'`, determining how
                data is saved when the dataset already exists.

        Notes:
            -   The file_path parameter can accept various types of storage locations
                depending on the zarr plugins installed. This includes local file
                paths, S3 URLs, etc.
            -   For cloud storage options, you may need to install additional
                dependencies such as s3fs for Amazon S3 access.
        """
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.approach = approach.strip().lower()

    def save(
        self, data: list[Any], indices: Any | None = None, **kwargs: dict[str, Any]
    ) -> None:
        """Saves the data to a Zarr array according to the specified approach.

        This method implements the abstract save method from the Saver base class.
        It persists the provided data to a Zarr array using the configured approach
        (append, overwrite, or update).

        The method handles creating new arrays if they don't exist (for append and
        overwrite approaches) or modifying existing arrays. It automatically converts
        the input data to a numpy array before saving.

        Args:
            data: A list of results to save. The data will be converted to a numpy
                array before saving to Zarr.
            indices: Required when approach is 'update', specifies the indices where
                data should be written in the existing array. Must match the shape
                and dimensionality of the input data.
            **kwargs: Additional keyword arguments passed to zarr.create_array or
                zarr.open_array. Common options include:

                - batches: Batch shape
                - dtype: Data type
                - compressor: Compression method (default: Blosc)
                - filters: Pre-compression filters

        Raises:
            ImportError: If the zarr library is not installed.
            ValueError: If approach is 'update' but indices is None.
            FileNotFoundError: If attempting to update a non-existent array.
            TypeError: If the data cannot be converted to a numpy array.

        Examples:
            Saving data with append approach:

            ```python
            saver = ZarrSaver("results.zarr")

            # First save creates the array
            saver.save([1, 2, 3, 4, 5])

            # Subsequent saves append to it
            saver.save([6, 7, 8, 9, 10])
            ```

            Saving with custom batch size and compression:

            ```python
            import numcodecs

            saver = ZarrSaver("compressed_results.zarr")

            # Save with customized storage parameters
            compressor = numcodecs.Blosc(cname="zstd", clevel=9)
            saver.save(large_dataset, batches=(1000,), compressor=compressor)
            ```

            Updating specific indices:

            ```python
            saver = ZarrSaver("timeseries.zarr", approach="update")

            # Update values at specific positions
            new_values = [99.5, 98.3, 97.8]
            indices = [10, 20, 30]  # Positions to update
            saver.save(new_values, indices=indices)
            ```

        Notes:
            -   The append operation is optimized for adding new data to existing
                arrays without reading the entire array into memory.
            -   For large datasets, consider specifying appropriate batch sizes
                in kwargs when creating the array for the first time.
            -   When updating, the indices and data must have compatible shapes.
            -   Unlike HDF5, zarr supports concurrent reads and writes from multiple
                processes, making it suitable for distributed computing environments.
        """
        if not has_zarr:
            raise ImportError("Zarr is not installed.")

        arr = np.array(data)

        store = zarr.storage.LocalStore(self.file_path, read_only=False)

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
