from typing import TYPE_CHECKING, Any, TypeVar

import ray

if TYPE_CHECKING:
    from raygent import Task

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


@ray.remote
def ray_worker(
    task_class: "Task[InputType, OutputType]",
    chunk: list[Any],
    at_once: bool,
    *args: tuple[Any],
    **kwargs: dict[str, Any],
) -> Any:
    """

    Remote Ray worker function that processes tasks in parallel.

    This function is a helper wrapper around a [`Task`][task.Task] object that executes
    the task's run method with the provided chunk of data. It serves as the core
    execution unit when using Ray for parallel processing within the `raygent`
    framework.

    While primarily used internally by [`TaskManager`][manager.TaskManager]'s
    [`_submit_ray`][manager.TaskManager._submit_ray] method, it can be
    called directly for custom Ray deployments if needed.

    Args:
        task_class: A callable that returns a [`Task`][task.Task] instance with
            [`run`][task.Task.run] and [`process_item`][task.Task.process_item]
            or [`process_items`][task.Task.process_items] methods.
        chunk: A list of items to be processed by the task.
        at_once: If `True`, processes all items at once using
            [`process_items`][task.Task.process_items];
            otherwise, processes each item individually using
            [`process_item`][task.Task.process_item].
        *args: Additional positional arguments passed to the task's
            [`run`][task.Task.run] method.
        **kwargs: Additional keyword arguments passed to the task's
            [`run`][task.Task.run] method. These can include task-specific parameters
            that customize execution.

    Returns:
        The results from executing the task's run method on the provided chunk.
            Typically this is a list of processed items or results.

    Examples:
        Basic usage through [`TaskManager`][manager.TaskManager] (recommended):

        ```python
        # This is handled automatically by TaskManager when use_ray=True
        manager = TaskManager(MyTask, use_ray=True)
        manager.submit_tasks(items)
        ```

        Direct usage (advanced):

        ```python
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Create a remote worker with 2 CPUs
        future = ray_worker.options(num_cpus=2).remote(
            MyTask, items_chunk, at_once=False, custom_param="value"
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
            at_once=True,
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
    task_instance = task_class()
    result = task_instance.run(chunk, at_once, *args, **kwargs)
    return result
