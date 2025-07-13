from typing import Any

import pytest

from raygent import Task, TaskManager


class DummySaver:
    """A simple saver to record saved data."""

    def __init__(self):
        self.saved_data = []

    def save(self, data):
        self.saved_data.append(data)


class DummyTask(Task[int, float]):
    """A dummy task that doubles the input."""

    def process_item(self, item: int, **kwargs: dict[str, Any]) -> float:
        return item * 2

    def process_items(self, items: list[int], **kwargs: dict[str, Any]) -> list[float]:
        return [item * 2 for item in items]


@pytest.fixture
def dummy_task_manager():
    # For testing, use DummyTask and sequential mode (use_ray=False).
    manager = TaskManager(task_class=DummyTask, use_ray=False)
    print(manager.result_handler)
    return manager


def test_submit_tasks_sequential(dummy_task_manager):
    items = [1, 2, 3, 4, 5]
    dummy_task_manager.submit_tasks(items, chunk_size=2, saver=None, at_once=False)
    results = dummy_task_manager.get_results()
    # Results may be stored as a list of lists. Flatten them.
    flattened = []
    for res in results:
        if isinstance(res, list):
            flattened.extend(res)
        else:
            flattened.append(res)
    assert flattened == [2, 4, 6, 8, 10]


def test_max_concurrent_tasks(dummy_task_manager):
    dummy_task_manager.n_cores = 8
    dummy_task_manager.n_cores_worker = 2
    assert dummy_task_manager.max_concurrent_tasks == 4


def test_submit_tasks_with_saver(dummy_task_manager):
    saver = DummySaver()
    items = [1, 2, 3, 4]
    dummy_task_manager.submit_tasks(
        items, chunk_size=2, saver=saver, at_once=False, save_interval=2
    )
    # Check that saver.saved_data is not empty.
    assert len(saver.saved_data) >= 1
