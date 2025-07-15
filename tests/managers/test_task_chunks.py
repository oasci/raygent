from typing import override

import pytest

from raygent import ResultsCollector, Task, TaskManager


class DummyTask(Task[list[float], list[float]]):
    """A dummy task that doubles the input."""

    @override
    def do(self, batch: list[float], *args: object, **kwargs: object) -> list[float]:
        return [item * 2 for item in batch]


@pytest.fixture
def dummy_task_manager():
    # For testing, use DummyTask and sequential mode (use_ray=False).
    result_handler = ResultsCollector[list[float]]()
    manager = TaskManager(
        task_cls=DummyTask, result_handler=result_handler, use_ray=False
    )
    return manager


def test_task_generator(dummy_task_manager):
    """
    Checks that the task generator works correctly based on batch size.
    """
    batch = list(range(10))
    batch_size = 3
    batches: list[tuple[int, list[int]]] = list(
        dummy_task_manager.batch_gen(batch, batch_size)
    )

    assert len(batches) == 4
    assert batches[0][0] == 0
    assert batches[0][1] == [0, 1, 2]
    assert batches[1][0] == 1
    assert batches[1][1] == [3, 4, 5]
    assert batches[2][0] == 2
    assert batches[2][1] == [6, 7, 8]
    assert batches[3][0] == 3
    assert batches[3][1] == [9]


def test_task_generator_large_batch(dummy_task_manager):
    """
    Checks that only one batch is returned when batch > len(batch)
    """
    batch = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch_size = 100
    batches: list[tuple[int, list[int]]] = list(
        dummy_task_manager.batch_gen(batch, batch_size)
    )
    assert len(batches) == 1
    assert batches[0][0] == 0
    assert batches[0][1] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
