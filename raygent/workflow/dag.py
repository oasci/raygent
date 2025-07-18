"""Minimalist DAG builder for deterministic, batch-synchronous stream processing."""

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload, override

from collections.abc import Iterable, Sequence

import ray

from raygent.results.handlers.handler import ResultsHandler
from raygent.workflow import BoundedQueue, NodeHandle, TaskActor

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
        self._nodes: dict[str, NodeHandle[Any]] = {}
        self._started: bool = False

    def add(
        self,
        task: "Task[T]",
        /,
        *,
        inputs: Iterable[NodeHandle[Any]] | None = None,
        name: str | None = None,
        handler: ResultsHandler[Any] | None = None,
        queue_size: int | None = None,
    ) -> NodeHandle[Any]:
        """Instantiate *task* as a `TaskActor` and connect it to *inputs*.

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

        parents: list[NodeHandle[Any]] = list(inputs or [])
        n_inputs: int = len(parents)
        actor = TaskActor.remote(task, n_inputs)

        # Wire edge queues between each parent and *this* new node.
        inbound_queues: "list[Queue]" = []
        qsize = queue_size or self._default_qsize
        for parent in parents:
            q = BoundedQueue[T](qsize)
            parent.actor.register_output.remote(q)
            parent.outputs.append(q)
            actor.register_input.remote(q)
            inbound_queues.append(q)

        if handler:
            actor.register_handler.remote(handler)
        handle = NodeHandle(actor=actor)
        key = name or f"{task.__class__.__name__}-{handle.uid}"
        if key in self._nodes:
            raise ValueError(f"Duplicate node name: {key!r}")
        self._nodes[key] = handle

        return handle

    @overload
    def add_source(
        self,
        task: "Task[T]",
        n_sources: Literal[1],
        /,
        *,
        name: str | None = None,
        handler: ResultsHandler[Any] | None = None,
        queue_size: int | None = None,
    ) -> tuple[NodeHandle[T], BoundedQueue[T]]: ...

    @overload
    def add_source(
        self,
        task: "Task[T]",
        n_sources: int,
        /,
        *,
        name: str | None = None,
        handler: ResultsHandler[Any] | None = None,
        queue_size: int | None = None,
    ) -> tuple[NodeHandle[T], list[BoundedQueue[T]]]: ...

    def add_source(
        self,
        task: "Task[T]",
        n_sources: int,
        /,
        *,
        name: str | None = None,
        handler: ResultsHandler[Any] | None = None,
        queue_size: int | None = None,
    ) -> tuple[NodeHandle[T], BoundedQueue[T] | list[BoundedQueue[T]]]:
        """Add a *root* operator and hand back its inbound queue.

        Returns (handle, queue) so you can `queue.put(...)` from the driver.
        """
        qsize = queue_size or self._default_qsize
        actor = TaskActor.remote(task, n_sources)
        sources = [BoundedQueue(qsize) for _ in range(n_sources)]
        for source in sources:
            actor.register_input.remote(source)

        if handler:
            actor.register_handler.remote(handler)
        handle = NodeHandle[T](actor=actor, inputs=sources)
        key = name or f"{task.__class__.__name__}-{handle.uid}"
        if key in self._nodes:
            raise ValueError(f"Duplicate node name: {key!r}")
        self._nodes[key] = handle
        if len(sources) == 1:
            sources = sources[0]
        return handle, sources

    def add_sink(
        self,
        task: "Task[T]",
        /,
        *,
        inputs: Iterable[NodeHandle[Any]] | None = None,
        name: str | None = None,
        handler: ResultsHandler[Any] | None = None,
        queue_size: int | None = None,
    ) -> tuple[NodeHandle[Any], BoundedQueue[Any]]:
        """Instantiate *task* as a `TaskActor` and connect it to *inputs*.

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

        parents: list[NodeHandle[Any]] = list(inputs or [])
        n_inputs: int = len(parents)
        actor = TaskActor.remote(task, n_inputs)

        inbound_queues: "list[Queue]" = []
        qsize = queue_size or self._default_qsize
        for parent in parents:
            q = BoundedQueue[T](qsize)
            parent.actor.register_output.remote(q)
            parent.outputs.append(q)
            actor.register_input.remote(q)
            inbound_queues.append(q)

        sink_queue: BoundedQueue[Any] = BoundedQueue[Any](qsize)
        actor.register_output.remote(sink_queue)

        if handler:
            actor.register_handler.remote(handler)
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
    def __repr__(self) -> str:  # pragma: no cover – cosmetic
        parts = ", ".join(self._nodes)
        return f"<DAG nodes=[{parts}] started={self._started}>"
