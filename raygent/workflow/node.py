from typing import TYPE_CHECKING, Any, Never, override

import uuid
from dataclasses import dataclass, field

import ray

from raygent.results import BatchMessage
from raygent.workflow import BoundedQueue

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from ray.util.queue import Queue

    from raygent import Task


@ray.remote
class TaskActor:
    def __init__(self, task: "Task", num_inputs: int) -> None:
        """
        Args:
            task: The [`Task`][task.Task] this Actor is responsible for.
            num_inputs: The number of inputs this `task` takes.
        """
        self.task: "Task" = task
        self.num_inputs: int = num_inputs
        self.input_queues: "list[Queue]" = []
        self.buffers: list[dict[int, Any]] = [{} for _ in range(num_inputs)]
        self.output_queues: "list[Queue]" = []

    def register_input(self, queue: "Queue") -> None:
        """Register a source queue for an input of `task`"""
        self.input_queues.append(queue)

    def register_output(self, queue: "Queue") -> None:
        """Register a sink queue for a `TaskActor` that consumes this output."""
        self.output_queues.append(queue)

    def run(self) -> Never:
        assert len(self.input_queues) == self.num_inputs

        while True:
            # 1) See if any index is “ready” (in all buffers).
            common_idxs = set(self.buffers[0].keys())
            for buf in self.buffers[1:]:
                common_idxs &= buf.keys()

            if common_idxs:
                # Process all ready indices in sorted order
                for idx in sorted(common_idxs):
                    batch = [buf.pop(idx) for buf in self.buffers]
                    result = self.task.do(*batch)
                    msg = BatchMessage(index=idx, payload=result)
                    for out_q in self.output_queues:
                        out_q.put(msg)
                continue

            # 2) If nothing is ready yet, fetch exactly one more message.
            #    Choose the input whose buffer is smallest to balance reads.
            sizes = [len(buf) for buf in self.buffers]
            i = sizes.index(min(sizes))
            # This will block only if that queue is empty; otherwise immediately
            msg = self.input_queues[i].get()
            self.buffers[i][msg.index] = msg.payload


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
