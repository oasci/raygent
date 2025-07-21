# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scienting Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.


from typing import Any, Generic, TypeVar

from abc import ABC, abstractmethod

T = TypeVar("T")


class Saver(ABC, Generic[T]):
    """Abstract base class for saving data in various formats or destinations.

    The Saver class provides a standardized interface for persisting computational
    results across the raygent framework. It abstracts away the details of how
    and where data is stored, allowing TaskRunner to work with different storage
    backends without modification.

    This design follows the Strategy pattern, where different concrete Saver
    implementations (strategies) can be interchanged to save data to files, databases,
    cloud storage, or other destinations using various formats and approaches.

    Custom Savers can be created by subclassing and implementing the save method.
    """

    @abstractmethod
    def save(
        self,
        data: T,
        indices: Any | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Saves the provided data to the configured destination.

        This abstract method must be implemented by all concrete Saver subclasses
        to define how data is written to the destination (file, database, cloud
        storage, etc.).

        The method should handle details such as format conversion, creating or
        opening the destination, writing the data, and handling any errors that
        may occur during the saving process.

        Args:
            data: A list of results to save. Each element can be of any type,
                though specific Saver implementations may have type requirements.
            indices: Optional indices or locations where data should be written
                when using an update approach. The meaning of this parameter depends
                on the specific Saver implementation. Default is None.
            **kwargs: Additional keyword arguments that may be used by specific
                saver implementations. These arguments are typically passed from
                TaskRunner.save_kwargs and can include parameters like data types,
                compression options, or format-specific settings.

        Returns:
            None. The effect of this method is to persist data to the
                configured destination.

        Raises:
            Implementation specific exceptions, such as IOError for file-based
                savers or connection errors for database savers.

        Examples:
            Implementation in HDF5Saver:

            ```python
            def save(self, data, indices=None, **kwargs):
                arr = np.array(data, dtype=kwargs.get("dtype"))

                with h5py.File(self.file_path, "a") as h5file:
                    if self.dataset_name not in h5file:
                        h5file.create_dataset(self.dataset_name, data=arr)
                    else:
                        dset = h5file[self.dataset_name]
                        old_size = dset.shape[0]
                        new_size = old_size + arr.shape[0]
                        dset.resize(new_size, axis=0)
                        dset[old_size:new_size] = arr
            ```

        Notes:
            - Implementations should be robust to repeated calls with new data.
            - They should appropriately handle the case where the destination
              does not yet exist vs. already exists.
            - The indices parameter is primarily used for update operations,
              and its interpretation varies between Saver implementations.
        """
