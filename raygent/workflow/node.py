from typing import TYPE_CHECKING, Any, TypeVar, override

import uuid
from dataclasses import dataclass, field

import ray
from ray.util.queue import Queue

from raygent.results import BatchMessage
from raygent.results.result import IndexedResult
from raygent.workflow import BoundedQueue

if TYPE_CHECKING:
    from raygent import Task
    from raygent.results.handlers import ResultsHandler

T = TypeVar("T")


@ray.remote
class TaskActor:
    def __init__(self, task: "Task[T]", num_inputs: int) -> None:
        """
        Args:
            task: The [`Task`][task.Task] this Actor is responsible for.
            num_inputs: The number of inputs this `task` takes.
        """
        self.task: "Task[T]" = task
        self.num_inputs: int = num_inputs
        self.input_queues: list[Queue] = []
        self.buffers: list[dict[int, Any]] = [{} for _ in range(num_inputs)]
        self.next_index: int = 0
        self.output_queues: list[Queue] = []
        self.handler: "ResultsHandler[T] | None" = None

    def register_input(self, queue: Queue) -> None:
        """Register a source queue for an input of `task`"""
        self.input_queues.append(queue)

    def register_output(self, queue: Queue) -> None:
        """Register a sink queue for a `TaskActor` that consumes this output."""
        self.output_queues.append(queue)

    def register_handler(self, handler: "ResultsHandler[T]") -> None:
        self.handler = handler

    def run(self):
        assert len(self.input_queues) == self.num_inputs

        while True:
            # Blocking until we have next_index in all input buffers
            for i, q in enumerate(self.input_queues):
                while self.next_index not in self.buffers[i]:
                    msg: BatchMessage[Any] = q.get()
                    self.buffers[i][msg.index] = msg.payload

            # Gather aligned batches
            batches = [buf.pop(self.next_index) for buf in self.buffers]

            # Execute task
            output = self.task.do(*batches)

            # Fan out to all downstream queues
            for out_q in self.output_queues:
                out_q.put(BatchMessage(index=self.next_index, payload=output))

            # Handle locally if final
            if self.handler:
                self.handler.add_result(
                    IndexedResult(value=output, index=self.next_index)
                )

            self.next_index += 1


@dataclass(slots=True, kw_only=True)
class NodeHandle:
    """Light‑weight handle around an instantiated `TaskActor`.

    Users primarily need this so they can reference the node as *input* when
    adding downstream tasks.
    """

    actor: ray.actor.ActorHandle
    outputs: list[BoundedQueue] = field(default_factory=list, repr=False)

    # Unique identifier mainly for debugging; *not* used as a dict key in DAG.
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8], init=False)

    @override
    def __repr__(self) -> str:  # pragma: no cover – cosmetic
        cname = type(self.actor).__name__
        return f"<NodeHandle {cname}[uid={self.uid}]>"
