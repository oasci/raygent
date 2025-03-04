from typing import Any
from abc import ABC

from loguru import logger


class Task(ABC):
    """Abstract base class for executing computational tasks on a collection of items.

    This class provides a structured way to process individual or multiple items using
    the `process_item` and `process_items` methods, respectively. Subclasses must
    implement these abstract methods to define specific task logic.

    Example:
        ```python
        >>> class MyTask(Task):
        ...     def process_item(self, item):
        ...         return item * 2
        ...
        >>> task = MyTask()
        >>> task.run([1, 2, 3])
        [2, 4, 6]
        ```
    """

    def run(self, items: list[Any], at_once: bool = False, **kwargs: dict[str, Any]) -> list[Any]:
        """Processes a list of items using the appropriate method.

        This method attempts to process each item individually using `process_item`
        unless `at_once` is set to `True`, in which case it processes all items
        together using `process_items`.

        Errors encountered during processing are logged and captured in the results.

        Args:
            items: A list of input data items to be processed.
            at_once: If `True`, calls `process_items` to process all
                items at once; otherwise, processes them individually.

        Returns:
            A list containing the results of processing each item, or an
                error tuple `("error", error_message)` in case of failures.

        Raises:
            TypeError: If `items` is not a list.
            Exception: Captures any exception raised during processing.

        Example:
            ```python
            >>> class SquareTask(Task):
            ...     def process_item(self, item):
            ...         return item ** 2
            ...     def process_items(self, items):
            ...         return [item ** 2 for item in items]
            ...
            >>> task = SquareTask()
            >>> task.run([1, 2, 3])
            [1, 4, 9]
            ```

        Note:
            - If `at_once=True`, processing efficiency may improve for batch operations.
            - Errors are logged using `loguru.logger`.
        """
        results = []
        if at_once:
            results = self.process_items(items, **kwargs)
        else:
            for item in items:
                try:
                    result = self.process_item(item, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.exception(f"Error processing item {item}: {e}")
                    results.append(("error", str(e)))
        return results

    def process_item(self, item: Any, **kwargs: dict[str, Any]) -> Any:
        """Processes a single item.

        This method must be implemented by subclasses to define the computation
        logic for an individual item.

        Args:
            item: The input data to be processed.

        Returns:
            The processed result.

        Raises:
            NotImplementedError: If called directly without implementation.

        Example:
            ```python
            >>> class DoubleTask(Task):
            ...     def process_item(self, item):
            ...         return item * 2
            ...
            >>> task = DoubleTask()
            >>> task.process_item(4)
            8
            ```
        """
        raise NotImplementedError

    def process_items(self, items: list[Any], **kwargs: dict[str, Any]) -> Any:
        """Processes multiple items at once.

        This method must be implemented by subclasses to define the batch processing logic.

        Args:
            items: A list of input data items to be processed.

        Returns:
            The processed result.

        Raises:
            NotImplementedError: If called directly without implementation.

        Example:
            ```python
            >>> class DoubleTask(Task):
            ...     def process_items(self, items):
            ...         return [item * 2 for item in items]
            ...
            >>> task = DoubleTask()
            >>> task.process_items([1, 2, 3, 4])
            [2, 4, 6, 8]
            ```
        """
        raise NotImplementedError
