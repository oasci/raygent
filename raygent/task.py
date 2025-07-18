from typing import Any, Generic, TypeVar

from abc import ABC, abstractmethod

from raygent.results import IndexedResult

T = TypeVar("T")


class Task(ABC, Generic[T]):
    """Protocol for executing computational tasks on collections of data.

    The `Task` class provides a flexible framework for processing data items and
    serves as the core computational unit in the `raygent` framework.

    **Types**

    To write a new [`Task`][task.Task], you need to first understand what your
    ParamSpecs and [`T`][dtypes.T] will be.
    ParamSpecs specify what positional and keyword arguments all methods will receive.

    Note that [`do`][task.Task.do] assumes a batch of multiple
    values will be provided. If your task is to square numbers, then you would provide
    something like:

    ```python
    SquareTask(Task[[list[float]], list[float]):
    ```

    If your task squeezes the data into a scalar (e.g., taking the sum), then you
    would specify the following.

    ```python
    SumTask(Task[[list[float]], float]):
    ```
    Performing operations on NumPy arrays are specified the same way, except now
    we get arrays for `InputType` and `T`.

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
        [`TaskRunner`][runner.TaskRunner] calls [`run_batch()`][task.Task.run_batch] to
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

    @abstractmethod
    def do(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> T | Exception:
        """Batch process data.

        This method defines the computation logic for batch processing a collection
        of data together. It is called by the [`run_batch`][task.Task.run_batch].

        Args:
            *args: Position arguments.
            **kwargs: Keyword arguments.

        Returns:
            The processed results for all data.

        Example:
            ```python
            class DocumentVectorizerTask(Task[str, list[dict[str, Any]]]):
                def do(self, batch, *args, **kwargs):
                    # Load model once for all documents
                    vectorizer = load_large_language_model()

                    # Process all documents in an optimized batch operation
                    # which is much faster than processing one at a time
                    embeddings = vectorizer.encode_batch(batch)

                    # Return results paired with original items
                    return [
                        {"document": doc, "vector": vec}
                        for doc, vec in zip(items, embeddings)
                    ]
            ```
        """
