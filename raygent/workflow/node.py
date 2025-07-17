from typing import Any, Generic

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto

from raygent import TaskRunner
from raygent.dtypes import BatchType
from raygent.results.handlers import HandlerType


class NodeStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass(slots=True, kw_only=True)
class RetryPolicy:
    max_retries: int = 0
    backoff_seconds: float = 0.0
    retry_exceptions: tuple[type[Exception], ...] = (Exception,)


@dataclass(slots=True)
class WorkflowNode(Generic[BatchType, HandlerType]):
    """
    A single vertex in a Workflow DAG. Each node owns (or can lazily create)
    one `TaskRunner` that performs the underlying work.
    """

    name: str
    runner: TaskRunner[BatchType, HandlerType]

    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    kwargs_task: Mapping[str, Any] = field(default_factory=dict)
    kwargs_remote: Mapping[str, Any] = field(default_factory=dict)

    status: NodeStatus = field(init=False, default=NodeStatus.PENDING)
    started_at: datetime | None = field(init=False, default=None)
    finished_at: datetime | None = field(init=False, default=None)
    duration: timedelta | None = field(init=False, default=None)
    attempt: int = field(init=False, default=0)
    error: Exception | None = field(init=False, default=None)

    results_ref: Any | None = field(init=False, default=None)
    input_ref: Any | None = field(init=False, default=None)

    def resolve_runner(self) -> TaskRunner[BatchType, HandlerType]:
        """Return the underlying runner.

        This is often used when validating types for edges.
        """
        return self.runner

    def mark_started(self) -> None:
        self.status, self.started_at = NodeStatus.RUNNING, datetime.now()

    def mark_finished(self) -> None:
        self.status = NodeStatus.COMPLETED
        self.finished_at = datetime.now()
        self.duration = self.finished_at - (self.started_at or self.finished_at)

    def mark_failed(self, exc: Exception) -> None:
        self.status = NodeStatus.FAILED
        self.error = exc
        self.finished_at = datetime.now()
        self.duration = self.finished_at - (self.started_at or self.finished_at)
