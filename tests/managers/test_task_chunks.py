from typing import Any

import pytest

from raygent import Task, TaskManager


class DummySaver:
    """A simple saver to record saved data."""

    def __init__(self):
        self.saved_data = []

    def save(self, data):
        self.saved_data.append(data)


class DummyTask(Task[float, float]):
    """A dummy task that doubles the input."""

    def process_item(self, item: float, **kwargs: dict[str, Any]) -> float:
        return item * 2

    def process_items(
        self, items: list[float], **kwargs: dict[str, Any]
    ) -> list[float]:
        return [item * 2 for item in items]


@pytest.fixture
def dummy_task_manager():
    # For testing, use DummyTask and sequential mode (use_ray=False).
    manager = TaskManager(task_class=DummyTask, use_ray=False)
    print(manager.result_handler)
    return manager


def test_task_generator(dummy_task_manager):
    """
    Checks that the task generator works correctly based on chunk size.
    """
    items = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    chunk_size = 3
    chunks = list(dummy_task_manager.task_generator(items, chunk_size))

    assert len(chunks) == 4
    assert chunks[0] == [0, 1, 2]
    assert chunks[1] == [3, 4, 5]
    assert chunks[2] == [6, 7, 8]
    assert chunks[3] == [9]


def test_task_generator_large_chunk(dummy_task_manager):
    """
    Checks that only one chunk is returned when chunk > len(items)
    """
    items = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    chunk_size = 100
    chunks = list(dummy_task_manager.task_generator(items, chunk_size))
    assert len(chunks) == 1
    assert chunks[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_task_generator_negative_chunk(dummy_task_manager):
    """ """
    items = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    chunk_size = -1
    chunks = list(dummy_task_manager.task_generator(items, chunk_size))
    assert len(chunks) == 0
