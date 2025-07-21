# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scienting Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.


from typing import TYPE_CHECKING, Any, Never, override

import uuid
from dataclasses import dataclass, field

import ray

from raygent.batch import BatchMessage
from raygent.workflow import BoundedQueue

if TYPE_CHECKING:
    from ray.actor import ActorHandle

    from raygent import Task


@dataclass
class QueueBuffer:
    max_size: int = 64
    """Maximum size allowed in buffer before we block queue get."""
    _buffer: dict[int, BatchMessage] = field(default_factory=dict)
    """Buffer of batch indices and their messages"""
    _indices: list[int] = field(default_factory=list)

    def add(self, message: BatchMessage) -> None:
        idx: int = message.index
        self._buffer[idx] = message
        self._indices.append(message.index)

    def pop(self, index: int) -> Any:
        _ = self._indices.remove(index)
        msg = self._buffer.pop(index)
        return msg.payload

    @property
    def indices(self) -> list[int]:
        return self._indices

    @property
    def size(self) -> int:
        return len(self._indices)


@ray.remote
class TaskActor:
    def __init__(
        self, task: "Task", num_inputs: int, task_kwargs: dict[str, Any] | None = None
    ) -> None:
        """
        Args:
            task: An initialized [`Task`][task.Task] this ray Actor is responsible for.
            num_inputs: The number of inputs the `task` takes. This is used
                to initialize [`buffers`][workflow.node.TaskActor.buffers].
            task_kwargs: Keyword arguments for [`do()`][task.Task.do] method. All
                positional arguments are handled with queues.
        """
        self.task: "Task" = task
        """An initialized [task][task.Task] this `TaskActor` will call when receiving
        data in [`input_queues`][workflow.node.TaskActor.input_queues]."""
        if task_kwargs is None:
            task_kwargs = {}
        self.task_kwargs: dict[str, Any] = task_kwargs

        self.input_queues: list[BoundedQueue] = []
        self.input_buffers: list[QueueBuffer] = []
        self.output_queues: list[BoundedQueue] = []

    def register_input(self, queue: "BoundedQueue", max_buffer_size: int = 64) -> None:
        """Register a source queue for an input of `task`"""
        self.input_queues.append(queue)
        self.input_buffers.append(QueueBuffer(max_buffer_size))

    def register_output(self, queue: "BoundedQueue") -> None:
        """Register a sink queue for this `TaskActor` that consumes this output."""
        self.output_queues.append(queue)

    def _get_ready_batches(self, sort: bool = True) -> list[int]:
        """Examine all input queue buffers and return a set of batch indices that
        are common across all input queues.

        Args:
            sort: Sort batch indices are from smallest to largest.

        Returns:
            Batch indices present in source queue buffers that are ready to process.
        """
        common_idxs: set[int] = set(self.input_buffers[0].indices)
        for buf in self.input_buffers[1:]:
            common_idxs &= set(buf.indices)
        ready_idxs: list[int] = list(common_idxs)
        if sort:
            ready_idxs = sorted(ready_idxs)
        return ready_idxs

    def _process_batch(self, idx: int) -> None:
        batch = tuple(buf.pop(idx) for buf in self.input_buffers)
        result = self.task.do(*batch, **self.task_kwargs)
        msg = BatchMessage(index=idx, payload=result)
        for out_q in self.output_queues:
            out_q.put(msg)

    def _buffer_next_message(self) -> None:
        """Get next message from input queues and put them in buffer.

        Will block the run loop until a message is received from the smallest buffer.
        """
        sizes = [buf.size for buf in self.input_buffers]
        i = sizes.index(min(sizes))
        msg: BatchMessage = self.input_queues[i].get()
        self.input_buffers[i].add(msg)

    def start(self) -> Never:
        """Start an infinite loop waiting for batch messages in input queues."""
        while True:
            ready_batch_idxs = self._get_ready_batches()
            if len(ready_batch_idxs) > 0:
                for idx in ready_batch_idxs:
                    self._process_batch(idx)
                continue
            self._buffer_next_message()


@dataclass(slots=True, kw_only=True)
class NodeHandle:
    """Light-weight handle around an instantiated `TaskActor`.

    Users primarily need this so they can reference the node as input when
    adding downstream tasks.
    """

    actor: "ActorHandle"
    inputs: list[BoundedQueue] = field(default_factory=list, repr=False)
    outputs: list[BoundedQueue] = field(default_factory=list, repr=False)

    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8], init=False)

    @override
    def __repr__(self) -> str:
        return f"<Node [uid={self.uid} in={self.inputs} out={self.outputs}]>"
