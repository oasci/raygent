# Tasks

A **task** in `raygent` represents a unit of computation that processes a batch of items using a user-defined class that extends [`RayTask`][task.RayTask].
Each task operates on a subset of input data and executes computations in parallel when using Ray, or sequentially if Ray is disabled.

The execution of a task follows a structured workflow, where the [`run`][task.RayTask.run] method orchestrates the computation by applying [`process_item`][task.RayTask.process_item] to each item in the batch.
This enables efficient parallelization while ensuring error handling and logging.
By defining a structured pipeline for task execution, users can efficiently scale computations across multiple cores or nodes, depending on their system capabilities and needs.

## `run`

The `run` method is responsible for processing multiple items at once. It determines whether to process each item individually using [`process_item`][task.RayTask.process_item] or to process all items together using [`process_items`][task.RayTask.process_items] when `at_once=True`. This flexibility allows users to optimize performance by reducing redundant computations in batch operations.

## `process_item`

The [`process_item`][task.RayTask.process_item] method defines the actual computation performed on a **single** item.
This allows tasks to be easily customized for different computational requirements, ranging from simple arithmetic operations to more complex data transformations.

**Example: Basic Numeric Computation**

```python
from raygent import RayTask


class SquareTask(RayTask):
    def process_item(self, item: float) -> float:
        return item ** 2

task = SquareTask()
print(task.run([1, 2, 3, 4]))  # Output: [1, 4, 9, 16]
```

## `process_items`

The [`process_items`][task.RayTask.process_items] method provides an efficient way to process multiple items together, reducing redundant computations that would otherwise be repeated in [`process_items`][task.RayTask.process_items].
This method is particularly useful when preprocessing steps can be shared among multiple items.

**Example: Computing the Mean of a List using NumPy**

```python
import numpy as np
import numpy.typing as npt
from raygent import RayTask


class MeanTask(RayTask):
    def process_items(self, items: npt.NDArray[np.float64]) -> np.float64:
        return np.mean(items)

task = MeanTask()
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(task.run(data, at_once=True))  # Output: 5.0
```

Using [task.RayTask.process_items] ensures optimized performance for numerical computations, making this approach well-suited for tasks involving large-scale data processing.
