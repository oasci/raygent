from typing import Any

from collections.abc import Callable

import ray


@ray.remote
def ray_worker(
    task_class: Callable[[], Any],
    chunk: list[Any],
    *args: tuple[Any],
    **kwargs: dict[str, Any],
) -> Any:
    """
    Remote function to process a single item using the provided task class.

    Args:
        task_class: A callable that returns an instance with a `run` method.
        item: The input data required for the calculation.

    Returns:
        Any: The result of the `run` method or an error tuple.
    """
    task_instance = task_class()
    result = task_instance.run(chunk, *args, **kwargs)
    return result
