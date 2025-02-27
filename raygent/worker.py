from typing import Any

from collections.abc import Callable

import ray


@ray.remote
def ray_worker(task_class: Callable[[], Any], chunk: list[Any]) -> Any:
    """
    Remote function to process a single item using the provided task class.

    Args:
        task_class (Callable[[], Any]): A callable that returns an instance with a `run` method.
        item (Any): The input data required for the calculation.

    Returns:
        Any: The result of the `run` method or an error tuple.
    """
    task_instance = task_class()
    result = task_instance.run(chunk)
    return result
