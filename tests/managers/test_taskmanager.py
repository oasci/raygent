from typing import Any

from raygent import ResultHandler, Task, TaskManager


class DummyTask(Task[list[float], list[float]]):
    """A dummy task that doubles the input."""

    def do(self, items: list[float], **kwargs: dict[str, Any]) -> list[float]:
        return [item * 2 for item in items]


def _flatten_results(results):
    flattened = []
    for res in results:
        if isinstance(res, list):
            flattened.extend(res)
        else:
            flattened.append(res)
    return flattened


def test_submit_tasks_sequential():
    items = [1, 2, 3, 4, 5]
    task_manager: TaskManager[list[float], list[float]] = TaskManager(
        DummyTask(), use_ray=False
    )

    task_manager.submit_tasks(items, chunk_size=2)
    result_handler: ResultHandler[list[float]] = task_manager.get_results()
    results = result_handler.get()
    flattened = _flatten_results(results)
    assert flattened == [2, 4, 6, 8, 10]


def test_submit_tasks_parallel():
    task_manager = TaskManager[list[float], list[float]](DummyTask(), use_ray=True)
    task_manager.n_cores = 2
    task_manager.n_cores_worker = 1

    items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    task_manager.submit_tasks(items, chunk_size=2)

    result_handler: ResultHandler[list[float]] = task_manager.get_results()
    results = result_handler.get()
    flattened = _flatten_results(results)

    assert task_manager.max_concurrent_tasks == 2
    assert flattened == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
