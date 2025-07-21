# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scienting Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.


"""Minimalist DAG builder for deterministic, batch-synchronous stream processing."""

from typing import TYPE_CHECKING, Any, TypeVar, override

import time
from collections.abc import Generator, Iterable, Sequence

import ray
from loguru import logger

from raygent.batch import BatchMessage, batch_generator
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
        inputs: NodeHandle | Iterable[NodeHandle],
        task_kwargs: dict[str, Any] | None = None,
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
                will send [`BatchMessage`][batch.BatchMessage]s to this
                `task`. If the [`Task`][task.Task] takes more than one positional
                argument, the order of these `inputs` matters.
            task_kwargs: Keyword arguments for [`do()`][task.Task.do] method. All
                positional arguments are handled with queues.
            name: Optional symbolic name. If omitted, a unique one is
                derived from the `task` class name and a UUID snippet.
            queue_size: Override for this node's incoming queues; defaults to
                the builder-level [`_default_qsize`][workflow.dag.DAG._default_qsize].
            n_workers: Maximum number of parallel workers for this node (i.e., the
                `max_concurrency` argument for ray).
            n_cpu_per_worker: Number of CPU cores per individual workers (i.e., the
                `num_cpus` argument for ray).

        Returns:
            A `NodeHandle` you can reference later when wiring children.
        """
        if self._started:
            raise RuntimeError("Cannot add nodes after DAG has been started")

        if task_kwargs is None:
            task_kwargs = {}

        if not isinstance(inputs, Iterable):
            inputs = (inputs,)

        parents: list[NodeHandle] = list(inputs or [])
        n_inputs: int = len(parents)
        actor = TaskActor.options(
            num_cpus=n_cpu_per_worker, max_concurrency=n_workers
        ).remote(task, n_inputs, task_kwargs=task_kwargs)

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
        input: NodeHandle,
        /,
        *,
        name: str | None = None,
        queue_size: int | None = None,
    ) -> BoundedQueue:
        """Create a sink where data can be retrieved.

        Args:
            input: An upstream `NodeHandle` objects.
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
        actor = TaskActor.options(num_cpus=1, max_concurrency=1).remote(task, 1)
        self.n_cores_requested += 2

        inbound_queues: "list[BoundedQueue]" = []
        qsize = queue_size or self._default_qsize

        q = BoundedQueue(qsize)
        input.actor.register_output.remote(q)
        input.outputs.append(q)

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

    def start(self) -> None:
        """Start the main loop (`.run()`) on every added `TaskActor`."""
        if self._started:
            raise RuntimeError("DAG already running")
        if not ray.is_initialized():
            raise RuntimeError("You must initialize ray first; not starting the DAG")
        self._started = True
        for handle in self._nodes.values():
            handle.actor.start.remote()

    def stream(
        self,
        *data_streams: Iterable[Any],
        source_queues: Sequence[BoundedQueue],
        sink_queues: Sequence[BoundedQueue],
        batch_size: int = 10,
        prebatched: bool = False,
        max_inflight: int = 16,
        sink_wait: float = 0.01,
    ) -> Generator[tuple[int, BatchMessage], Any, None]:
        """Stream data into DAG through source queues and yields from sinks."""
        if not self._started:
            raise RuntimeError("DAG must be running before streaming")

        if sink_wait <= 0.0:
            raise ValueError("sink_wait must be greater than 0.0")

        batch_gen = batch_generator(
            *data_streams, batch_size=batch_size, prebatched=prebatched
        )
        sent_batches = 0
        yielded_messages = 0
        num_sinks = len(sink_queues)
        sent_all = False

        def get_any() -> tuple[int, BatchMessage]:
            # try non‐blocking on all queues
            while True:
                for i, q in enumerate(sink_queues):
                    try:
                        msg = q.get(block=False)
                        return i, msg
                    except Exception:
                        continue
                # nothing ready yet, back off briefly
                time.sleep(sink_wait)

        while not (sent_all and yielded_messages == sent_batches * num_sinks):
            # send as many as we can
            while (
                not sent_all
                and (sent_batches - (yielded_messages // num_sinks)) < max_inflight
            ):
                try:
                    idx, payloads = next(batch_gen)
                except StopIteration:
                    sent_all = True
                    break
                for q, p in zip(source_queues, payloads):
                    q.put(BatchMessage(index=idx, payload=p))
                sent_batches += 1

            # drain one result from ANY sink
            queue_idx, out_msg = get_any()
            yield queue_idx, out_msg
            yielded_messages += 1

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
