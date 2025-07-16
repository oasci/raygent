from typing import Any

from collections.abc import Callable, Iterable

try:
    import ray

    has_ray = True
except ImportError:
    has_ray = False

from loguru import logger

from raygent import TaskManager
from raygent.savers import Saver

TaskManagerFactory = Callable[[], TaskManager[Any, Any]]
TaskManagerOrFactory = TaskManager[Any, Any] | TaskManagerFactory


class WorkflowTaskNode:
    """
    Represents a single task node within the workflow graph.
    Encapsulates task-specific configuration and state.
    """

    def __init__(
        self,
        name: str,
        manager_or_factory: TaskManagerOrFactory,
        output_to_disk: bool = False,
        output_saver: Saver | None = None,
        output_chunk_size: int | None = None,
        task_specific_kwargs_task: dict[str, Any] | None = None,
        task_specific_kwargs_remote: dict[str, Any] | None = None,
    ):
        self.name: str = name
        self.manager_or_factory: TaskManagerOrFactory = manager_or_factory
        self.output_to_disk: bool = output_to_disk
        self.output_saver: Saver | None = output_saver
        self.output_chunk_size: int | None = output_chunk_size
        self.task_specific_kwargs_task: dict[str, Any] = task_specific_kwargs_task or {}
        self.task_specific_kwargs_remote: dict[str, Any] = (
            task_specific_kwargs_remote or {}
        )

        # Runtime state
        self.status: str = "pending"  # pending, ready, running, completed, failed
        self.error: str | None = None
        # Using | for Union and ray.ObjectRef requires importing ray if available
        self.results: "list[Any] | ray.ObjectRef | str | None" = None
        self.input_data: "Iterable[Any] | ray.ObjectRef | None" = None

    def get_task_manager_instance(self) -> TaskManager:
        """Instantiate or return the TaskManager for this node."""
        if callable(self.manager_or_factory):
            return self.manager_or_factory()
        return self.manager_or_factory

    def update_status(self, status: str, error: str | None = None):
        """Update the status of the task node."""
        self.status = status
        self.error = error
        logger.debug(f"Task '{self.name}' status updated to: {status}")

