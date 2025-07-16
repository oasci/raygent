# Manager

[`TaskManager`][manager.TaskManager] is a class designed to efficiently manage task execution using **Ray's Task Parallelism**.
It allows for seamless parallel execution of tasks while also supporting sequential execution when Ray is disabled.
By handling task submission, worker management, and result collection, [`TaskManager`][manager.TaskManager] simplifies distributed processing workflows.
Some key features include:

-   **Flexible Execution Modes**: Runs tasks either sequentially or in parallel using Ray.
-   **Dynamic Resource Allocation**: Automatically detects available CPU cores unless specified.
-   **Task Batching**: Supports processing data in configurable batch sizes for optimized parallelism.
-   **Asynchronous Execution with Ray**: Manages Ray object references (`futures`) for efficient workload distribution.
-   **Custom Save Function Support**: Allows intermediate results to be saved at specified intervals.

## Initialization

The [`TaskManager`][manager.TaskManager] constructor initializes the execution environment and prepares task handling.

```python
from raygent import TaskManager
from raygent.results.handlers import ResultsCollector
from my_tasks import ExampleTask

manager = TaskManager(ExampleTask, ResultsCollector, n_cores=4, use_ray=True)
```

## Submitting Tasks

Tasks are submitted using [`submit_tasks`][manager.TaskManager.submit_tasks], which divides data into batches and manages workers efficiently.

```python
manager.submit_tasks(data=[1, 2, 3, 4, 5], batch_size=2)
```
