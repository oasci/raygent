from raygent import Task, TaskRunner
from raygent.results.handlers import ResultsCollector


class DummyTask(Task[list[float], list[float]]):
    """A dummy task that doubles the input."""

    def do(self, batch: list[float], *args: object, **kwargs: object) -> list[float]:
        return [item * 2 for item in batch]


def _flatten_results(results):
    flattened = []
    for res in results:
        if isinstance(res, list):
            flattened.extend(res)
        else:
            flattened.append(res)
    return flattened


def test_submit_tasks_sequential():
    batch = [1.0, 2.0, 3.0, 4.0, 5.0]
    task_runner: TaskRunner[list[float], ResultsCollector[list[float]]] = TaskRunner(
        DummyTask, ResultsCollector, in_parallel=False
    )

    handler = task_runner.submit_tasks(batch, batch_size=2)
    results = handler.get()
    flattened = _flatten_results(results)
    assert flattened == [2.0, 4.0, 6.0, 8.0, 10.0]


def test_submit_tasks_parallel():
    task_runner = TaskRunner[list[float], ResultsCollector[list[float]]](
        DummyTask, ResultsCollector, in_parallel=True
    )
    task_runner.n_cores = 2
    task_runner.n_cores_worker = 1

    batch = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    handler = task_runner.submit_tasks(batch, batch_size=2)

    results = handler.get()
    flattened = _flatten_results(results)

    assert task_runner.max_concurrent_tasks == 2
    assert flattened == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
