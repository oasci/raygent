from typing import Any, Protocol, TypeVar

from raygent.results import Result

InputType = TypeVar(name="InputType", contravariant=True)
OutputType = TypeVar(name="OutputType")


class Task(Protocol[InputType, OutputType]):
    """Protocol for executing computational tasks on collections of data.

    The `Task` class provides a flexible framework for processing data items and
    serves as the core computational unit in the `raygent` framework.

    This class implements the Template Method pattern, where the base class (`Task`)
    defines the protocol of an algorithm in its [`run`][task.Task.run] method,
    while deferring some steps to subclasses through the
    [`process_items`][task.Task.process_items] method.

    Notes:
        -   The Task class is designed to be used with the
            [`TaskManager`][manager.TaskManager] for parallel execution across multiple
            cores or machines.
        -   When used with Ray (via [`TaskManager`][manager.TaskManager]), each
            Task instance will be created on the worker node, so any initialization
            in `__init__` will happen per worker.
    """

    def __init__(self) -> None:
        super().__init__()

    def setup(self, **kwargs: dict[str, Any]) -> None:
        """Optional setup method called once before processing begins."""

    def teardown(self, **kwargs: dict[str, Any]) -> None:
        """Optional teardown method called once after all processing is complete."""

    def run(
        self,
        items: tuple[int, InputType],
        **kwargs: dict[str, Any],
    ) -> Result[OutputType]:
        """This method serves as the primary entry point for task execution.

        Args:
            items: The chunk index and items to be processed.
            *args: Addutional positional arguments pass to
                [`startup`][task.Task.startup],
                [`process_items`][task.Task.process_items], and
                [`teardown`][task.Task.teardown].
            **kwargs: Additional keyword arguments passed to
                [`startup`][task.Task.startup],
                [`process_items`][task.Task.process_items], and
                [`teardown`][task.Task.teardown].

        Returns:
            Results of processing all data in `items`.

        Examples:

            ```python
            class NumberSquarerTask(Task[float, float]):
                def process_items(self, items):
                    return [i**2 for i in items]


            task = NumberSquarerTask()
            results = task.run([1.0, 2.0, 3.0, 4.0, 5.0])
            ```

            Batch processing:

            ```python
            class VectorMultiplierTask(Task[npt.NDArray[np.float64], list[float]]):
                def process_items(self, items, **kwargs):
                    scale = kwargs.get("scale", 1.0)
                    return (arr * scale).tolist()


            items = np.array([1, 2, 3, 4, 5], dtype=np.float64)
            task = VectorMultiplierTask()
            results = task.run(items, scale=2.0)
            ```
        """
        self.setup(**kwargs)
        assert isinstance(items[0], int), "items needs to be a tuple of index, "
        result: Result[OutputType] = Result[OutputType](index=items[0])
        output: OutputType | Exception = self.process_items(items[1], **kwargs)
        if isinstance(output, Exception):
            result.error = output
        else:
            result.value = output

        self.teardown(**kwargs)
        return result

    def process_items(
        self,
        items: InputType,
        **kwargs: dict[str, Any],
    ) -> OutputType | Exception:
        """Processes multiple items at once in a batch operation.

        This method defines the computation logic for batch processing a collection
        of items together. It is called by the [`run`][task.Task.run].

        Args:
            items: Data to be processed together.
            **kwargs: Additional keyword arguments that customize processing behavior.
                These arguments are passed directly from the `run` method.

        Returns:
            The processed results for all items, typically a list matching the input
                length, but could be any structure depending on the implementation.

        Raises:
            NotImplementedError: If the class does not implement this method.

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
        raise NotImplementedError
