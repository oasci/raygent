from typing import TYPE_CHECKING

import ray

if TYPE_CHECKING:
    from raygent import Task
    from raygent.results import IndexedResult

from raygent.dtypes import BatchType, OutputType


@ray.remote
def ray_worker(
    task_cls: "type[Task[BatchType, OutputType]]",
    index: int,
    batch: BatchType,
    *args: object,
    **kwargs: object,
) -> "IndexedResult[OutputType]":
    """
    Remote Ray worker function that processes tasks in parallel.

    This function is a wrapper around a [`Task`][task.Task] object that executes
    the task's [`run_batch`][task.Task.run_batch] method with the provided batch
    of data. It serves as the core execution unit when using Ray.

    While primarily used internally by [`TaskManager`][manager.TaskManager]'s
    [`_submit_ray`][manager.TaskManager._submit_ray] method, it can be
    called directly for custom Ray deployments if needed.

    Args:
        task_cls: A class that is type [`Task`][task.Task].
        index: Batch index used for [`Result.index`][results.result.Result.index].
        batch: `InputType` to be processed by the task.
        *args: Additional positional arguments passed to the task's
            [`run_batch`][task.Task.run_batch] method.
        **kwargs: Additional keyword arguments passed to the task's
            [`run_batch`][task.Task.run_batch] method. These can include task-specific parameters
            that customize execution.

    Returns:
        The results from executing the task's run_batch method on the provided batch.
            Typically this is a list of processed batch or results.

    Examples:
        Basic usage through [`TaskManager`][manager.TaskManager] (recommended):

        ```python
        # This is handled automatically by TaskManager when use_ray=True
        manager = TaskManager(MyTask, ResultsCollector, use_ray=True)
        manager.submit_tasks(batch)
        ```

        Direct usage (advanced):

        ```python
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Create a remote worker with 2 CPUs
        future = ray_worker.options(num_cpus=2).remote(
            MyTask, index, batch, custom_param="value"
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
            index,
            large_batch,
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
    task: "Task[BatchType, OutputType]" = task_cls()
    return task.run_batch(index, batch, *args, **kwargs)
