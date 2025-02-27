from typing import Any
from abc import ABC, abstractmethod

from loguru import logger


class RayTask(ABC):
    """A parent class of others that governs what calculations are run on each
    task."""

    def run(self, items: list[Any]) -> list[Any]:
        """Processes multiple items.

        This method wraps the `process_item` method with a try-except block.

        Args:
            items: The input data required for the calculation.

        Returns:
            The result of the calculations or an error tuple.
        """
        results = []
        for item in items:
            try:
                result = self.process_item(item)
                results.append(result)
            except Exception as e:
                logger.exception(f"Error processing item {item}: {e}")
                results.append(("error", str(e)))
        return results

    @abstractmethod
    def process_item(self, item: Any) -> Any:
        """The definition that computes or processes a single item.

        This method should be implemented by subclasses to define the specific
        processing logic for each item.

        Args:
            item: The input data required for the calculation.

        Returns:
            Any: The result of processing the item.
        """
