# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scienting Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.


from typing import Any

from abc import ABC, abstractmethod


class Task(ABC):
    """Protocol for executing computational tasks on collections of data.

    The `Task` class provides a flexible framework for processing data items and
    serves as the core computational unit in the `raygent` framework.

    The only required implementation is [`do`][task.Task.do]
    which specifies how to process a batch of items. For example, writing a
    [`Task`][task.Task] that computes the mean value across rows of a
    NumPy array could be implemented like so.

    ```python
    from raygent import Task
    import numpy as np
    import numpy.typing as npt


    class MeanTask(Task):
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
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def do(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any | Exception:
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
