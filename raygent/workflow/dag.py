"""Minimalist DAG builder for deterministic, batch-synchronous stream processing."""

from typing import TYPE_CHECKING, Any, Generator, TypeVar, Unpack, override

from collections.abc import Iterable, Sequence

import ray
from loguru import logger

from raygent.batch import batch_generator
from raygent.results.result import BatchMessage
from raygent.workflow import BoundedQueue, NodeHandle, TaskActor
from raygent.workflow.helpers import IdentityTask

if TYPE_CHECKING:
    from ray.actor import ActorHandle

    from raygent import Task

T = TypeVar("T")


class DAG:
    """Builder / orchestrator for a deterministic, queue-backed execution graph from
    [tasks][task.Task].

    The builder is _stateful_. You add tasks one by one, and the class wires
    ray queues and actors as you go. Once all nodes are created, call
    [`run`][workflow.dag.DAG.run] to start every actor's main loop concurrently waiting
    for messages in the source-node queues.
    """

    def __init__(self, *, queue_size: int = 128, max_cores: int = 8) -> None:
        """Create an empty DAG builder with no nodes.

        Args:
            queue_size: Default `maxsize` for every edge queue.
        """

        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self._default_qsize: int = queue_size
        self._nodes: dict[str, NodeHandle] = {}
        self._edges: dict[str, BoundedQueue] = {}
        self._started: bool = False
        self.max_cores: int = max_cores
        self.n_cores_requested: int = 0
        """Number of logical CPU cores requested for all tasks and queues."""

    def check_n_cores(self) -> None:
        if self.n_cores_requested > self.max_cores:
            logger.warning(
                "Total number of requested cores ({}) is greater than `self.max_cores` ({})",
                self.n_cores_requested,
                self.max_cores,
            )

    def add(
        self,
        task: "Task",
        /,
        *,
        inputs: Iterable[NodeHandle],
        name: str | None = None,
        queue_size: int | None = None,
        n_workers: int = 1,
        n_cpu_per_worker: int = 1,
    ) -> NodeHandle:
        """Add a [`Task`][task.Task] that will be fully connected to the `DAG`. In
        other words, no user-facing queues to send or recieve messages directly. All
        inbound and outbond messages are other nodes, including
        [sources][workflow.dag.DAG.add_source] and
        [sinks][workflow.dag.DAG.add_sink].

        Args:
            task: An instance of your [`Task`][task.Task] API. This minimally should
                have a [`do()`][task.Task.do] implementation.
            inputs: Upstream [`NodeHandle`][workflow.node.NodeHandle] objects that
                will send [`BatchMessage`][results.result.BatchMessage]s to this
                `task`. If the [`Task`][task.Task] takes more than one positional
                argument, the order of these `inputs` matters.
            name: Optional symbolic name. If omitted, a unique one is
                derived from the `task` class name and a UUID snippet.
            queue_size: Override for this node's incoming queues; defaults to
                the builder-level [`_default_qsize`][workflow.dag.DAG._default_qsize].
            n_workers: Maximum number of parallel workers for this node.

        Returns:
            A `NodeHandle` you can reference later when wiring children.
        """
        if self._started:
            raise RuntimeError("Cannot add nodes after DAG has been started")

        parents: list[NodeHandle] = list(inputs or [])
        n_inputs: int = len(parents)
        actor = TaskActor.options(
            num_cpus=n_cpu_per_worker, max_concurrency=n_workers
        ).remote(task, n_inputs)

        self.n_cores_requested += n_workers

        # Wire queues between each parent and this new node.
        inbound_queues: "list[BoundedQueue]" = []
        qsize = queue_size or self._default_qsize
        for parent in parents:
            q = BoundedQueue(qsize)

            # head
            parent.actor.register_output.remote(q)
            parent.outputs.append(q)

            # tail
            actor.register_input.remote(q)
            inbound_queues.append(q)

            self._edges[q.uid] = q
            self.n_cores_requested += 2  # one for each parent queue

        handle = NodeHandle(actor=actor, inputs=inbound_queues)

        key_node = name or f"{task.__class__.__name__}-{handle.uid}"
        if key_node in self._nodes:
            raise ValueError(f"Duplicate node name: {key_node!r}")
        self._nodes[key_node] = handle

        self.check_n_cores()

        logger.info("Successfully created node: {}", repr(handle))

        return handle

    def add_source(
        self,
        /,
        *,
        name: str | None = None,
        queue_size: int | None = None,
    ) -> tuple[NodeHandle, BoundedQueue]:
        """Add a source node that broadcasts batches to all output nodes.

        Args:
            name: Provide a human-readable name instead of a random hash.
            queue_size: Maximum number of messages in flight before adding
                back pressure.

        Returns:
            A [`NodeHandle`][workflow.node.NodeHandle] for referencing this source
                in other nodes.
            The [bounded queue][workflow.queue.BoundedQueue] used to inject batches
                into the `DAG` through this source node.

        Examples:
            No inputs are needed to create a source node.

            ```python
            dag = DAG()
            sn_1, sq_1 = dag.add_source()  # First source_node and source_queue
            ```
        """
        if self._started:
            raise RuntimeError("Cannot add nodes after DAG has been started")

        task = IdentityTask()
        qsize = queue_size or self._default_qsize
        actor = TaskActor.options(num_cpus=1, max_concurrency=1).remote(task, 1)
        self.n_cores_requested += 1

        source = BoundedQueue(qsize)
        actor.register_input.remote(source)
        self._edges[source.uid] = source

        handle = NodeHandle(actor=actor, inputs=[source])

        key = name or f"{task.__class__.__name__}-{handle.uid}"
        if key in self._nodes:
            raise ValueError(f"Duplicate node name: {key!r}")
        self._nodes[key] = handle

        self.check_n_cores()

        logger.info("Successfully created source node: {}", repr(handle))

        return handle, source

    def add_sink(
        self,
        inputs: Iterable[NodeHandle],
        /,
        *,
        name: str | None = None,
        queue_size: int | None = None,
    ) -> BoundedQueue:
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
        actor = TaskActor.options(num_cpus=1, max_concurrency=1).remote(task, n_inputs)
        self.n_cores_requested += 2

        inbound_queues: "list[BoundedQueue]" = []
        qsize = queue_size or self._default_qsize
        for parent in parents:
            q = BoundedQueue(qsize)
            parent.actor.register_output.remote(q)
            parent.outputs.append(q)
            actor.register_input.remote(q)
            inbound_queues.append(q)

            self._edges[q.uid] = q

        sink_queue: BoundedQueue = BoundedQueue(qsize)
        actor.register_output.remote(sink_queue)

        handle = NodeHandle(actor=actor, inputs=inbound_queues, outputs=[sink_queue])
        key = name or f"{task.__class__.__name__}-{handle.uid}"
        if key in self._nodes:
            raise ValueError(f"Duplicate node name: {key!r}")
        self._nodes[key] = handle

        self.check_n_cores()

        logger.info("Successfully created sink node: {}", repr(handle))

        return sink_queue

    def run(self) -> None:
        """Start the main loop (`.run()`) on every added `TaskActor`."""
        if self._started:
            raise RuntimeError("DAG already running")
        self._started = True
        for handle in self._nodes.values():
            handle.actor.run.remote()

    def stream(
        self,
        *data_streams: Iterable[Any],
        source_queues: Sequence[BoundedQueue],
        sink_queues: Sequence[BoundedQueue],
        batch_size: int = 10,
        prebatched: bool = False,
        max_inflight: int = 16,
    ) -> Generator[tuple[int, BatchMessage], Any, None]:
        """Stream data into DAG through source queues and yields from sinks."""
        if not self._started:
            raise RuntimeError("DAG must be running before streaming")

        batch_gen = batch_generator(
            *data_streams, batch_size=batch_size, prebatched=prebatched
        )
        in_flight = 0
        sent_all = False

        def get_any() -> tuple[int, BatchMessage]:
            # try non‐blocking on all queues
            for i, q in enumerate(sink_queues):
                try:
                    msg = q.get(block=False)
                    return i, msg
                except Exception:
                    pass
            # If none ready, block on the first one
            msg = sink_queues[0].get()
            return 0, msg

        while not sent_all or in_flight > 0:
            # send as many as we can
            while not sent_all and in_flight < max_inflight:
                try:
                    idx, payloads = next(batch_gen)
                except StopIteration:
                    sent_all = True
                    break
                for q, p in zip(source_queues, payloads):
                    q.put(BatchMessage(index=idx, payload=p))
                in_flight += 1

            # drain one result from ANY sink
            queue_idx, out_msg = get_any()
            yield queue_idx, out_msg
            in_flight -= 1

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
