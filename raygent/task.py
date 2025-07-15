from typing import Generic

from abc import ABC, abstractmethod

from raygent.dtypes import BatchType, OutputType
from raygent.results import Result


class Task(ABC, Generic[BatchType, OutputType]):
    """Protocol for executing computational tasks on collections of data.

    The `Task` class provides a flexible framework for processing data items and
    serves as the core computational unit in the `raygent` framework.

    This class implements the Template Method pattern, where the base class (`Task`)
    defines the protocol of an algorithm in its [`run_batch`][task.Task.run_batch] method,
    while deferring some steps to subclasses through the
    [`do`][task.Task.do] method.

    **Types**

    To write a new [`Task`][task.Task], you need to first understand what your
    `InputType` and `OutputType` will be. These types specify what data that
    [`do`][task.Task.do] will receive in `items` and
    expected to return.
    Note that [`items`][task.Task.do] assumes a batch of multiple values
    will be provided. If your task is to square numbers, then you would provide
    something like:

    ```python
    SquareTask(Task[list[float], list[float]):
    ```

    If your task squeezes the data into a scalar (e.g., taking the sum), then you
    would specify the following.

    ```python
    SumTask(Task[list[float], float]):
    ```
    Performing operations on NumPy arrays can be specified the same way, except now
    we get arrays for `InputType` and `OutputType`.

    ```python
    SquareTask(Task[npt.NDArray[np.float64], npt.NDArray[np.float64]):
    ```

    **Implementation**

    The only required implementation is [`do`][task.Task.do]
    which specifies how to process a batch of items. For example, writing a
    [`Task`][task.Task] that computes the mean value across rows of a
    NumPy array could be implemented like so.

    ```python
    from raygent import Task
    import numpy as np
    import numpy.typing as npt

    class MeanTask(Task[npt.NDArray[np.float64], npt.NDArray[np.float64]):
        def do(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            mean = np.mean(items, axis=1)
            return mean
    ```

    We can just call [`do`][task.Task.do] directly if we
    want to use this task.

    ```python
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    task = MeanTask()
    mean = task.do(arr)  # Returns: np.array([2., 5.])
    ```

    !!! note
        [`TaskManager`][manager.TaskManager] calls [`run_batch()`][task.Task.run_batch] to
        produce a [`Result`][results.Result] to handle any setup, teardown, and
        errors.

        You can get the same data from our `task` by accessing
        [`Result.value`][results.Result.value].

        ```python
        index = 0
        result = task.run_batch(index, arr)
        result.value  # Returns: np.array([2., 5.])
        ```

    """

    def __init__(self) -> None:
        super().__init__()

    def setup(self, *args: object, **kwargs: object) -> None:
        """Optional setup method called once before processing begins."""

    def teardown(self, *args: object, **kwargs: object) -> None:
        """Optional teardown method called once after all processing is complete."""

    def run_batch(
        self,
        index: int,
        batch: BatchType,
        *args: object,
        **kwargs: object,
    ) -> Result[OutputType]:
        """This method serves as the primary entry point for task execution.

        Args:
            index: A unique integer used to specify ordering for
                [`Result.index`][results.Result.index].
            batch: Data to process using [`do`][task.Task.do].
            *args: Additional positional arguments pass to
                [`startup`][task.Task.startup],
                [`do`][task.Task.do], and
                [`teardown`][task.Task.teardown].
            **kwargs: Additional keyword arguments passed to
                [`startup`][task.Task.startup],
                [`do`][task.Task.do], and
                [`teardown`][task.Task.teardown].

        Returns:
            Results of processing all data in `items`.

        Example:
            ```python
            class NumberSquarerTask(Task[float, float]):
                def do(self, items):
                    return [i**2 for i in items]


            task = NumberSquarerTask()
            result = task.run_batch(0, [1.0, 2.0, 3.0, 4.0, 5.0])
            ```
        """
        self.setup(**kwargs)
        result: Result[OutputType] = Result[OutputType](index=index)
        output: OutputType | Exception = self.do(batch, *args, **kwargs)
        if isinstance(output, Exception):
            result.error = output
        else:
            result.value = output

        self.teardown(**kwargs)
        return result

    @abstractmethod
    def do(
        self,
        batch: BatchType,
        *args: object,
        **kwargs: object,
    ) -> OutputType | Exception:
        """Processes multiple items at once in a batch operation.

        This method defines the computation logic for batch processing a collection
        of items together. It is called by the [`run_batch`][task.Task.run_batch].

        Args:
            batch: Data to be processed together.
            **kwargs: Additional keyword arguments that customize processing behavior.
                These arguments are passed directly from the `run_batch` method.

        Returns:
            The processed results for all items, typically a list matching the input
                length, but could be any structure depending on the implementation.

        Raises:
            NotImplementedError: If the class does not implement this method.

        Example:
            ```python
            class DocumentVectorizerTask(Task[str, list[dict[str, Any]]]):
                def do(self, items, **kwargs):
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
