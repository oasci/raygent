from typing import Any

import pytest

from raygent import ResultHandler, Task, TaskManager


class DummyTask(Task[list[float], list[float]]):
    """A dummy task that doubles the input."""

    def process_items(
        self, items: list[float], **kwargs: dict[str, Any]
    ) -> list[float]:
        return [item * 2 for item in items]


@pytest.fixture
def dummy_task_manager():
    # For testing, use DummyTask and sequential mode (use_ray=False).
    result_handler = ResultHandler()
    manager = TaskManager(task=DummyTask, result_handler=result_handler, use_ray=False)
    return manager


def test_task_generator(dummy_task_manager):
    """
    Checks that the task generator works correctly based on chunk size.
    """
    items = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    chunk_size = 3
    chunks: list[tuple[int, list[int]]] = list(
        dummy_task_manager.task_generator(items, chunk_size)
    )

    assert len(chunks) == 4
    assert chunks[0][0] == 0
    assert chunks[0][1] == [0, 1, 2]
    assert chunks[1][0] == 1
    assert chunks[1][1] == [3, 4, 5]
    assert chunks[2][0] == 2
    assert chunks[2][1] == [6, 7, 8]
    assert chunks[3][0] == 3
    assert chunks[3][1] == [9]


def test_task_generator_large_chunk(dummy_task_manager):
    """
    Checks that only one chunk is returned when chunk > len(items)
    """
    items = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    chunk_size = 100
    chunks: list[tuple[int, list[int]]] = list(
        dummy_task_manager.task_generator(items, chunk_size)
    )
    assert len(chunks) == 1
    assert chunks[0][0] == 0
    assert chunks[0][1] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
