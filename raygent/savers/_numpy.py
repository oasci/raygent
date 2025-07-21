# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scienting Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.


from typing import Any, Literal

import os

from raygent.savers import Saver

try:
    import numpy as np

    has_numpy = True
except ImportError:
    has_numpy = False


class NumpySaver(Saver):
    """A saver that writes data to a .npy file.

    `NumpySaver` provides functionality to persist computational results in NumPy's
    .npy format, which is optimized for storing and loading numpy arrays. This format
    preserves shape, data type, and other array information, making it ideal for
    numerical data.

    This implementation supports three approaches to saving data:

    - `append`: Add new data to the existing array (creates if not exists)
    - `overwrite`: Replace existing data with new data
    - `update`: Update specific indices in the existing array with new values

    NumPy's .npy format offers specific advantages:

    - Fast load and save operations
    - Preservation of data types and array structures
    - Compact binary storage
    - Native integration with NumPy's ecosystem

    Requirements:
        This saver requires the numpy package to be installed:

        ```sh
        pip install numpy
        ```

    Examples:
        Basic usage:

        ```python
        # Create a NumpySaver for storing results
        saver = NumpySaver("results.npy")

        # Use with TaskRunner
        task_runner = TaskRunner(MyTask, ResultsCollector)
        task_runner.submit_tasks(batch, saver=saver, save_interval=100)
        ```

        Overwriting existing data:

        ```python
        # Create a saver that overwrites existing data
        saver = NumpySaver("daily_metrics.npy", approach="overwrite")

        # Save new results, replacing any existing file
        results = process_batch(today_data)
        saver.save(results)
        ```

        Updating specific indices:

        ```python
        # Create a saver for updating existing data
        saver = NumpySaver("time_series.npy", approach="update")

        # Update specific indices with new values
        new_data = [99.5, 98.3, 97.8]
        indices = [10, 20, 30]  # Positions to update
        saver.save(new_data, indices=indices)
        ```

    Notes:
        - NumPy's .npy format is best suited for numerical data where the entire
          array structure needs to be preserved.
        - For very large datasets where memory is a concern, consider using HDF5Saver
          or ZarrSaver instead, as .npy files are loaded entirely into memory.
        - The append operation loads the entire existing array into memory before
          appending, which may be inefficient for very large arrays.
        - For multidimensional arrays, shape compatibility is important when using
          append or update approaches.
    """

    def __init__(
        self,
        file_path: str,
        approach: Literal["append", "overwrite", "update"] = "append",
    ):
        """Initialize a NumpySaver instance.

        Args:
            file_path: The path to the .npy file where data will be saved.
            approach: One of `append`, `overwrite`, or `update`, determining how data
                is saved when the file already exists.

        Notes:
            - The file_path should have the .npy extension for compatibility with
              NumPy's load and save functions.
            - The approach parameter determines the behavior when saving data to an
              existing file:

              -  `append`: Concatenates new data to existing data
              -  `overwrite`: Replaces the entire file with new data
              -  `update`: Modifies specific indices in the existing data
        """
        self.file_path = file_path
        self.approach = approach.strip().lower()

    def save(
        self, data: list[Any], indices: Any | None = None, **kwargs: dict[str, Any]
    ) -> None:
        """Saves the data to a .npy file according to the specified approach.

        This method implements the abstract save method from the Saver base class.
        It persists the provided data to a NumPy .npy file using the configured approach
        (append, overwrite, or update).

        The method handles creating new files, appending to existing files, or
        updating specific indices in existing files. It automatically converts
        the input data to a numpy array before saving.

        Args:
            data: A list of results to save. The data will be converted to a numpy
                array before saving.
            indices: Required when approach is 'update', specifies the indices where
                data should be written in the existing array. Must be compatible with
                the shape of the input data.
            **kwargs: Additional keyword arguments. Current implementation does not
                use these parameters, but they are accepted for compatibility with
                the Saver interface.

        Raises:
            ImportError: If the numpy library is not installed.
            ValueError: If approach is 'update' but indices is None.
            ValueError: If an unknown approach is specified.
            FileNotFoundError: If attempting to update a non-existent file.
            TypeError: If the data cannot be converted to a numpy array.

        Examples:
            Saving data with the append approach:

            ```python
            saver = NumpySaver("results.npy")

            # First save creates the file
            saver.save([1, 2, 3, 4, 5])

            # Subsequent saves append to it
            saver.save([6, 7, 8, 9, 10])
            ```

            Overwriting existing data:

            ```python
            saver = NumpySaver("metrics.npy", approach="overwrite")

            # Save data, replacing any existing file
            saver.save([10, 20, 30, 40, 50])
            ```

            Updating specific indices:

            ```python
            saver = NumpySaver("values.npy", approach="update")

            # First create the file
            saver.approach = "overwrite"
            saver.save([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            # Then update specific positions
            saver.approach = "update"
            new_values = [99, 88, 77]
            indices = [2, 5, 8]  # Positions to update
            saver.save(new_values, indices=indices)
            # Result would be [0, 0, 99, 0, 0, 88, 0, 0, 77, 0]
            ```

        Notes:
            - The append operation loads the entire existing file into memory,
              concatenates the new data, and saves the combined result. This
              may be inefficient for very large arrays.
            - When updating, the indices and data must have compatible shapes.
            - For large datasets, consider using HDF5Saver or ZarrSaver which
              have more efficient append and update operations.
        """
        if not has_numpy:
            raise ImportError("NumPy is not installed.")

        arr = np.array(data)

        # If overwriting, simply save
        if self.approach == "overwrite":
            np.save(self.file_path, arr)

        elif self.approach == "append":
            if os.path.exists(self.file_path):
                data = np.load(self.file_path)
                combined = np.concatenate([data, arr])
                np.save(self.file_path, combined)
            else:
                np.save(self.file_path, arr)
        elif self.approach == "update":
            if os.path.exists(self.file_path):
                data = np.load(self.file_path)
                if not indices:
                    raise ValueError("`indices` cannot be None when using 'update'")
                data[indices] = arr
                np.save(self.file_path, data)
        else:
            raise ValueError(
                f"Unknown approach '{self.approach}'. Use 'append', 'update', or 'overwrite'."
            )
