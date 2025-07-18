"""Minimalist DAG builder for deterministic, batch-synchronous stream processing."""

from typing import TYPE_CHECKING, TypeVar, override

from collections.abc import Iterable, Sequence

import ray

from raygent.workflow import BoundedQueue, NodeHandle, TaskActor
from raygent.workflow.helpers import IdentityTask

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from ray.util.queue import Queue

    from raygent import Task

T = TypeVar("T")


class DAG:
    """Builder / orchestrator for a deterministic, queue-backed execution graph.

    The builder is *stateful*. You add tasks one by one, and the class wires
    queues & actors as you go. Once you've added everything, call
    `run` to start every actor's main loop concurrently.
    """

    def __init__(self, *, queue_size: int = 128) -> None:
        """Create an empty DAG builder.

        Args:
            queue_size: Default *maxsize* for every edge queue.
        """

        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self._default_qsize: int = queue_size
        self._nodes: dict[str, NodeHandle] = {}
        self._started: bool = False

    def add(
        self,
        task: "Task",
        /,
        *,
        inputs: Iterable[NodeHandle],
        name: str | None = None,
        queue_size: int | None = None,
    ) -> NodeHandle:
        """Instantiate *task* as a `TaskActor` and connect it to *inputs*.

        Args:
            task: An implementation of your [`Task`][task.Task] API.
            inputs: Zero or more upstream `NodeHandle` objects.
            name: Optional symbolic name. If omitted, a unique one is
                derived from the task class and a UUID snippet.
            queue_size: Override for this node's incoming queues; defaults to
                the builder-level `queue_size`.

        Returns:
            A `NodeHandle` you can reference later when wiring children.
        """

        if self._started:
            raise RuntimeError("Cannot add nodes after DAG has been started")

        parents: list[NodeHandle] = list(inputs or [])
        n_inputs: int = len(parents)
        actor = TaskActor.remote(task, n_inputs)

        # Wire edge queues between each parent and *this* new node.
        inbound_queues: "list[Queue]" = []
        qsize = queue_size or self._default_qsize
        for parent in parents:
            q = BoundedQueue(qsize)
            parent.actor.register_output.remote(q)
            parent.outputs.append(q)
            actor.register_input.remote(q)
            inbound_queues.append(q)

        handle = NodeHandle(actor=actor)
        key = name or f"{task.__class__.__name__}-{handle.uid}"
        if key in self._nodes:
            raise ValueError(f"Duplicate node name: {key!r}")
        self._nodes[key] = handle

        return handle

    def add_source(
        self,
        /,
        *,
        name: str | None = None,
        queue_size: int | None = None,
    ) -> tuple[NodeHandle, BoundedQueue]:
        """Add a *root* operator and hand back its inbound queue.

        Returns (handle, queue) so you can `queue.put(...)` from the driver.
        """
        task = IdentityTask()
        qsize = queue_size or self._default_qsize
        actor = TaskActor.remote(task, 1)

        source = BoundedQueue(qsize)
        actor.register_input.remote(source)

        handle = NodeHandle(actor=actor, inputs=[source])
        key = name or f"{task.__class__.__name__}-{handle.uid}"
        if key in self._nodes:
            raise ValueError(f"Duplicate node name: {key!r}")
        self._nodes[key] = handle

        return handle, source

    def add_sink(
        self,
        inputs: Iterable[NodeHandle],
        /,
        *,
        name: str | None = None,
        queue_size: int | None = None,
    ) -> tuple[NodeHandle, BoundedQueue]:
        """Create a sink where data can be retrieved.

        Args:
            task: An implementation of your [`Task`]task.Task] API.
            sources: Zero or more source queues to attach.
            inputs: Zero or more upstream `NodeHandle` objects.
            name: Optional symbolic name. If omitted, a unique one is
                derived from the task class and a UUID snippet.
            queue_size: Override for *this node's* incoming queues; defaults to
                the builder-level ``queue_size``.

        Returns:
            A `NodeHandle` you can reference later when wiring children.
        """

        if self._started:
            raise RuntimeError("Cannot add nodes after DAG has been started")

        task = IdentityTask()
        parents: list[NodeHandle] = list(inputs or [])
        n_inputs: int = len(parents)
        actor = TaskActor.remote(task, n_inputs)

        inbound_queues: "list[Queue]" = []
        qsize = queue_size or self._default_qsize
        for parent in parents:
            q = BoundedQueue(qsize)
            parent.actor.register_output.remote(q)
            parent.outputs.append(q)
            actor.register_input.remote(q)
            inbound_queues.append(q)

        sink_queue: BoundedQueue = BoundedQueue(qsize)
        actor.register_output.remote(sink_queue)

        handle = NodeHandle(actor=actor)
        key = name or f"{task.__class__.__name__}-{handle.uid}"
        if key in self._nodes:
            raise ValueError(f"Duplicate node name: {key!r}")
        self._nodes[key] = handle

        return handle, sink_queue

    def run(self) -> None:
        """Start the *main loop* (`.run()`) on every added ``TaskActor``."""
        if self._started:
            raise RuntimeError("DAG already running")
        self._started = True
        for handle in self._nodes.values():
            handle.actor.run.remote()

    def stop(self) -> None:
        """Kill every actor in the DAG (best effort)."""
        for handle in self._nodes.values():
            try:
                ray.kill(handle.actor, no_restart=True)
            except Exception:
                pass
        self._started = False

    def actor_handles(self) -> "Sequence[ActorHandle]":
        """Return a *read-only* view of every underlying actor handle."""
        return tuple(h.actor for h in self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    @override
    def __repr__(self) -> str:
        parts = ", ".join(self._nodes)
        return f"<DAG nodes=[{parts}] started={self._started}>"
