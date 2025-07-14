from typing import TYPE_CHECKING, Any, TypeVar

from collections.abc import Iterable

import ray

if TYPE_CHECKING:
    from raygent import Task

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


@ray.remote
def ray_worker(
    task: "Task[InputType, OutputType]",
    index: int,
    chunk: InputType | Iterable[InputType],
    **kwargs: dict[str, Any],
) -> OutputType:
    """

    Remote Ray worker function that processes tasks in parallel.

    This function is a helper wrapper around a [`Task`][task.Task] object that executes
    the task's run_chunk method with the provided chunk of data. It serves as the core
    execution unit when using Ray for parallel processing within the `raygent`
    framework.

    While primarily used internally by [`TaskManager`][manager.TaskManager]'s
    [`_submit_ray`][manager.TaskManager._submit_ray] method, it can be
    called directly for custom Ray deployments if needed.

    Args:
        task: A callable that returns a [`Task`][task.Task] instance with
            [`run_chunk`][task.Task.run_chunk] and [`process_item`][task.Task.process_item]
            or [`do`][task.Task.do] methods.
        index: Chunk index.
        chunk: A list of items to be processed by the task.
        *args: Additional positional arguments passed to the task's
            [`run_chunk`][task.Task.run_chunk] method.
        **kwargs: Additional keyword arguments passed to the task's
            [`run_chunk`][task.Task.run_chunk] method. These can include task-specific parameters
            that customize execution.

    Returns:
        The results from executing the task's run_chunk method on the provided chunk.
            Typically this is a list of processed items or results.

    Examples:
        Basic usage through [`TaskManager`][manager.TaskManager] (recommended):

        ```python
        # This is handled automatically by TaskManager when use_ray=True
        manager = TaskManager(MyTask(), use_ray=True)
        manager.submit_tasks(items)
        ```

        Direct usage (advanced):

        ```python
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Create a remote worker with 2 CPUs
        future = ray_worker.options(num_cpus=2).remote(
            MyTask, items_chunk, custom_param="value"
        )

        # Get results
        results = ray.get(future)
        ```

        Advanced configuration:

        ```python
        # Configure worker with custom resources and retries
        future = ray_worker.options(
            num_cpus=4, num_gpus=1, max_retries=3, resources={"custom_resource": 1}
        ).remote(
            ComplexTask,
            large_chunk,
            preprocessing_steps=["normalize", "filter"],
            batch_size=64,
        )
        ```

    Note:
        All Ray options (`num_cpus`, `num_gpus`, etc.) should be specified via the
        `options()` method on the `ray_worker` function, not as direct arguments.

        This function is decorated with `@ray.remote`, making it a Ray remote function
        that can be executed on any worker in the Ray cluster.
    """
    return task.run_chunk(index, chunk, **kwargs)
