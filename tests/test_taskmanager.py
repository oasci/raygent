import pytest

from raygent import TaskManager
from raygent import Task


class DummySaver:
    """A simple saver to record saved data."""

    def __init__(self):
        self.saved_data = []

    def save(self, data):
        self.saved_data.append(data)


class DummyTask(Task):
    """A dummy task that doubles the input."""

    def process_item(self, item, **kwargs):
        return item * 2

    def process_items(self, items, **kwargs):
        return [item * 2 for item in items]


@pytest.fixture
def dummy_task_manager(DummyTask):
    # For testing, use DummyTask and sequential mode (use_ray=False).
    manager = TaskManager(task_class=DummyTask, use_ray=False)
    # Ensure the result_handler has an empty results list.
    manager.result_handler.results = []
    return manager


def test_task_generator(dummy_task_manager):
    items = list(range(10))
    chunk_size = 3
    chunks = list(dummy_task_manager.task_generator(items, chunk_size))
    assert len(chunks) == 4
    assert chunks[0] == [0, 1, 2]
    assert chunks[-1] == [9]


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
