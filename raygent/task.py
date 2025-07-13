from typing import Any, Generic, TypeVar

from abc import ABC
from collections.abc import Collection

from loguru import logger

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class Task(ABC, Generic[InputType, OutputType]):
    """Abstract base class for executing computational tasks on collections of data.

    The `Task` class provides a flexible framework for processing data items either
    individually or in batches. It serves as the core computational unit in the
    `raygent` framework, defining how work is performed on data items.

    This class implements the Template Method pattern, where the base class (`Task`)
    defines the skeleton of an algorithm in its [`run`][task.Task.run] method,
    while deferring some steps to subclasses through the abstract
    [`process_item`][task.Task.process_item] and
    [`process_items`][task.Task.process_items] methods.

    Subclasses must implement at least one of the processing methods:

    -   [`process_item`][task.Task.process_item]: For item-by-item processing
        (more flexible, better error isolation)
    -   [`process_items`][task.Task.process_items]: For batch processing
        (potentially more efficient for shared resources)

    The choice between these methods depends on the specific requirements of the task,
    such as performance needs, error handling requirements, and interdependencies
    between items.

    Attributes:
        No class-level attributes are defined. All state should be managed within
        subclass instances.

    Examples:
        Basic implementation example:

        ```python
        class TextAnalyzerTask(Task[str, dict[str, Any]]):
            def __init__(self):
                self.nlp = load_nlp_model()  # Load model once per task instance

            def process_item(self, item, **kwargs):
                # Process each text individually with options from kwargs
                min_word_length = kwargs.get("min_word_length", 3)
                return {
                    "text": item,
                    "word_count": len(
                        [w for w in item.split() if len(w) >= min_word_length]
                    ),
                    "sentiment": self.nlp.analyze_sentiment(item),
                }


        # Usage
        analyzer = TextAnalyzerTask()
        results = analyzer.run(
            ["Hello world", "Goodbye wonderful world"], min_word_length=4
        )
        ```

        Implementing both processing methods:

        ```python
        class DataProcessorTask(Task[npt.NDArray[np.generic], dict[str, Any]]):
            def process_item(self, item, **kwargs):
                # For processing single items with detailed error handling
                try:
                    return self._transform_data(item, **kwargs)
                except ValueError as e:
                    return {"item": item, "error": str(e), "status": "failed"}

            def process_items(self, items, **kwargs):
                # For batch processing when efficiency is critical
                import numpy as np

                # Convert to numpy array for vectorized operations
                data = np.array(items)
                processed = self._batch_transform(data, **kwargs)
                return processed.tolist()

            def _transform_data(self, item, **kwargs):
                # Helper method with common transformation logic
                pass

            def _batch_transform(self, data_array, **kwargs):
                # Helper method with vectorized transformation logic
                pass
        ```

        Working with TaskManager:

        ```python
        from raygent.manager import TaskManager


        # Create a custom task
        class MyTask(Task[float, float]):
            def process_item(self, item, **kwargs):
                return item * 2


        # Process items using TaskManager (handles parallelization)
        manager = TaskManager(MyTask, use_ray=True, n_cores=4)
        manager.submit_tasks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        results = manager.get_results()
        ```

    Notes:
        -   The Task class is designed to be used with the TaskManager for parallel
            execution across multiple cores or machines.
        -   When used with Ray (via [`TaskManager`][manager.TaskManager]), each
            Task instance will be created
            on the worker node, so any initialization in `__init__` will happen
            per worker.
        -   Error handling is automatically provided for individual item processing,
            but must be manually implemented for batch processing.
        -   The design allows for flexible task implementation while maintaining a
            consistent interface for the execution framework.
    """

    def setup(self, **kwargs: dict[str, Any]) -> None:
        """
        Optional setup method called once before processing begins.
        Useful for setting up connections or loading shared resources.
        """

    def teardown(self, **kwargs: dict[str, Any]) -> None:
        """
        Optional teardown method called once after all processing is complete.
        Useful for closing connections or releasing resources.
        """

    def run(
        self,
        items: Collection[InputType],
        at_once: bool = False,
        **kwargs: dict[str, Any],
    ) -> list[OutputType | tuple[str, str]]:
        """Processes a list of items using either individual or batch processing.

        This method serves as the primary entry point for task execution.
        It determines whether to process items individually or as a batch based on the
        `at_once` parameter, and then delegates to either
        [`process_item`][task.Task.process_item] or
        [`process_items`][task.Task.process_items] accordingly.

        The method handles error management, ensuring that failures in processing
        individual items don't halt the entire execution. When operating in
        individual mode (`at_once=False`), errors for specific items are captured
        and included in the results list.

        Args:
            items: A list of input data items to be processed.
            at_once: If `True`, processes all items together using
                [`process_items`][task.Task.process_items];
                otherwise, processes each item individually using
                [`process_item`][task.Task.process_item].
            **kwargs: Additional keyword arguments passed to either
                [`process_item`][task.Task.process_item] or
                [`process_items`][task.Task.process_items] depending on which
                method is called.

        Returns:
            A list containing the results of processing each item. When `at_once=False`,
                errors are represented as tuples in the form `("error", error_message)`.
                When `at_once=True`, the entire output of
                [`process_items`][task.Task.process_items] is returned.

        Raises:
            TypeError: If `items` is not a list.
            NotImplementedError: Indirectly, if the appropriate processing method
                ([`process_item`][task.Task.process_item] or
                [`process_items`][task.Task.process_items]) has not been implemented.

        Notes:
            Choose `at_once=False` (default) when:

            - Items are independent and can be processed separately
            - You need granular error handling for each item
            - Memory constraints require processing one item at a time
            - You want to easily parallelize across items

            Choose `at_once=True` when:

            - There's significant shared setup or resource loading
            - Batch operations are more efficient (e.g., with vectorized libraries)
            - Items have interdependencies in their processing
            - The overhead of processing each item separately is high

        Examples:
            Individual processing (default):

            ```python
            class NumberSquarerTask(Task[float, float]):
                def process_item(self, item, **kwargs):
                    return item**2


            task = NumberSquarerTask()
            results = task.run([1.0, 2.0, 3.0, 4.0, 5.0])
            # results = [1., 4., 9., 16., 25.]
            ```

            Batch processing:

            ```python
            class VectorMultiplierTask(Task[npt.NDArray[np.generic], list[float]]):
                def process_items(self, items, **kwargs):
                    # Apply scaling factor from kwargs if provided
                    scale = kwargs.get("scale", 1.0)
                    return (arr * scale).tolist()


            task = VectorMultiplierTask()
            results = task.run([1, 2, 3, 4, 5], at_once=True, scale=2.0)
            # results = [2.0, 4.0, 6.0, 8.0, 10.0]
            ```

            Error handling with individual processing:

            ```python
            class DivisionTask(Task[float, float]):
                def process_item(self, item, **kwargs):
                    divisor = kwargs.get("divisor", 2)
                    return item / divisor


            task = DivisionTask()
            results = task.run([4, 6, 0, 8], divisor=0)
            # results may contain error tuples for division by zero
            # e.g., [('error', 'division by zero'), ('error', 'division by zero'),
            #        ('error', 'division by zero'), ('error', 'division by zero')]
            ```
        """
        self.setup(**kwargs)
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
        self.teardown(**kwargs)
        return results

    def process_item(self, item: InputType, **kwargs: dict[str, Any]) -> OutputType:
        """Processes a single item independently.

        This method defines the computation logic for processing an individual item.
        It is called by the `run` method when `at_once=False` (the default), allowing
        items to be processed one at a time.

        Subclasses must implement this method to define specific task logic, especially
        when items can be processed independently without shared context or resources.

        Args:
            item: A single data item to be processed.
            **kwargs: Additional keyword arguments that customize processing behavior.
                These arguments are passed directly from the `run` method.

        Returns:
            The processed result for the single item.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Notes:
            Choose this implementation approach when:

            - Items can be processed completely independently.
            - There is minimal shared setup or resource loading.
            - You want fine-grained error handling per item.
            - Processing can be easily parallelized.

            This method is typically more memory-efficient for large datasets as
            it processes one item at a time, but may incur repeated setup costs
            if each item requires loading the same resources.

        Example:
            ```python
            class TextClassifierTask(Task[str, dict[str, Any]]):
                def __init__(self):
                    # This setup happens once per task instance
                    self.model = load_classification_model()

                def process_item(self, item, **kwargs):
                    # Each text is processed independently
                    return {
                        "text": item,
                        "category": self.model.predict(item),
                        "confidence": self.model.confidence(item),
                    }
            ```
        """
        raise NotImplementedError(
            "process_item was requested with at_once=False, but not implemented"
        )

    def process_items(
        self, items: Collection[InputType], **kwargs: dict[str, Any]
    ) -> list[OutputType]:
        """Processes multiple items at once in a batch operation.

        This method defines the computation logic for batch processing a collection
        of items together. It is called by the [`run`][task.Task.run] method when
        `at_once=True`, allowing for optimized batch operations.

        Subclasses must implement this method to define specific batch processing
        logic, especially when there are substantial efficiency gains from processing
        items together or when items share common setup or resources.

        Args:
            items: A list of data items to be processed together.
            **kwargs: Additional keyword arguments that customize processing behavior.
                These arguments are passed directly from the `run` method.

        Returns:
            The processed results for all items, typically a list matching the input
                length, but could be any structure depending on the implementation.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Notes:
            Choose this implementation approach when:

            - Substantial shared setup or resource loading is required.
            - Vectorized operations can improve performance.
            - Items benefit from being processed in context with each other.
            - Memory overhead of loading all items at once is acceptable.

            This method can be significantly more efficient when:

            - Loading common files, models, or resources that would otherwise
                be loaded repeatedly in process_item
            - Using libraries that have optimized batch operations (like NumPy,
                PyTorch, or TensorFlow)
            - Items have interdependencies in their processing logic.

        Example:
            ```python
            class DocumentVectorizerTask(Task[str, list[dict[str, Any]]]):
                def process_items(self, items, **kwargs):
                    # Load model once for all documents
                    vectorizer = load_large_language_model()

                    # Process all documents in an optimized batch operation
                    # which is much faster than processing one at a time
                    embeddings = vectorizer.encode_batch(items)

                    # Return results paired with original items
                    return [
                        {"document": doc, "vector": vec}
                        for doc, vec in zip(items, embeddings)
                    ]
            ```
        """
        raise NotImplementedError(
            "process_items was requested with at_once=True, but not implemented"
        )
