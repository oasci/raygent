# Tasks

A **task** in `raygent` represents a unit of computation that processes a batch of items using a user-defined class that extends [`Task`][task.Task].
Each task operates on a subset of input data and executes computations in parallel when using Ray, or sequentially if Ray is disabled.

The execution of a task follows a structured workflow, where the [`run_batch`][task.Task.run_batch] method orchestrates the computation by applying [`do`][task.Task.do] to each item in the batch.
This enables efficient parallelization while ensuring error handling and logging.
By defining a structured pipeline for task execution, users can efficiently scale computations across multiple cores or nodes, depending on their system capabilities and needs.

## `do`

The [`do`][task.Task.do] method provides an efficient way to process multiple items together, reducing redundant computations that would otherwise be repeated in [`do`][task.Task.do].
This method is particularly useful when preprocessing steps can be shared among multiple items.

**Example: Computing the Mean of a NumPy array**

```python
import numpy as np
import numpy.typing as npt
from raygent import Task


class MeanTask(Task[np.float64]):
    def do(self, batch: npt.NDArray[np.float64]) -> np.float64:
        return np.mean(items)

task = MeanTask()
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(task.do(data))  # Output: 5.0
```

Using [do][task.Task.do] ensures optimized performance for numerical computations, making this approach well-suited for tasks involving large-scale data processing.

## `run_batch`

The `run_batch` method is responsible for processing multiple items at once.
