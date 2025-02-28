from abc import ABC, abstractmethod
from typing import Any


class Saver(ABC):
    """
    Abstract base class for saving data in various formats or destinations.

    Subclasses should implement the `save` method to define how data
    is written to disk, a database, or any other storage.
    """

    @abstractmethod
    def save(
        self, data: list[Any], indices: Any | None = None, **kwargs: dict[str, Any]
    ) -> None:
        """
        Saves the provided data.

        Args:
            data: A list of results to save.
            **kwargs: Additional keyword arguments that may be used by specific saver
                implementations.
        """
        pass
