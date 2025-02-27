# Manager

[`RayManager`][manager.RayManager] is a class designed to efficiently manage task execution using **Ray's Task Parallelism**.
It allows for seamless parallel execution of tasks while also supporting sequential execution when Ray is disabled.
By handling task submission, worker management, and result collection, [`RayManager`][manager.RayManager] simplifies distributed processing workflows.
Some key features include:

-   **Flexible Execution Modes**: Runs tasks either sequentially or in parallel using Ray.
-   **Dynamic Resource Allocation**: Automatically detects available CPU cores unless specified.
-   **Task Chunking**: Supports processing data in configurable chunk sizes for optimized parallelism.
-   **Asynchronous Execution with Ray**: Manages Ray object references (`futures`) for efficient workload distribution.
-   **Custom Save Function Support**: Allows intermediate results to be saved at specified intervals.

## Initialization

The [`RayManager`][manager.RayManager] constructor initializes the execution environment and prepares task handling.

```python
from raygent.manager import RayManager
from my_tasks import ExampleTask

ray_manager = RayManager(ExampleTask, n_cores=4, use_ray=True)
```

## Submitting Tasks

Tasks are submitted using [`submit_tasks`][manager.RayManager.submit_tasks], which divides data into chunks and manages workers efficiently.

```python
ray_manager.submit_tasks(
    items=[1, 2, 3, 4, 5],
    chunk_size=2,
    save_func=my_save_function,
    save_interval=2
)
```

## Retrieving Results

Returns a list of all collected results.

```python
results = ray_manager.get_results()
```
