from typing import Any

import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait

from loguru import logger

from raygent.batch import batch_generator
from raygent.workflow.executors import get_executor
from raygent.workflow.graph import WorkflowGraph
from raygent.workflow.node import WorkflowNode


def _run_task_runner(
    node: WorkflowNode[Any, Any],
    payload: Iterable[Any],
) -> Any:
    """Submit *batch* to this node's TaskRunner and block for the result."""
    tm = node.resolve_runner()

    node.mark_started()
    handler = tm.submit_tasks(
        payload,  # positional inputs
        batch_size=1,
        prebatched=True,
        kwargs_task=node.kwargs_task,  # now contains only the named inputs
        # args_remote=node.args_remote,
        kwargs_remote=node.kwargs_remote,
    )
    result = handler.get()
    node.results_ref = result
    node.mark_finished()
    return result


class _NodeRunner:
    """
    Wraps a WorkflowNode instance to enforce a concurrency limit and
    book-keep in-flight tasks.
    """

    def __init__(self, node: WorkflowNode[Any, Any], concurrency: int):
        self.node = node
        self.concurrency = max(1, concurrency)
        self.inflight: int = 0  # tasks currently submitted

    def can_accept(self) -> bool:
        return self.inflight < self.concurrency

    def submit(
        self,
        payload: Iterable[Any],
        executor: Executor,
    ) -> Future:
        self.inflight += 1

        def _wrapper() -> Any:
            try:
                return _run_task_runner(self.node, payload)
            finally:
                self.inflight -= 1

        return executor.submit(_wrapper)


class WorkflowRunner:
    """
    Simple, synchronous DAG scheduler. It streams batches from *sources*
    through the WorkflowGraph, honouring per-node concurrency limits.
    """

    def __init__(
        self,
        graph: WorkflowGraph,
        *,
        default_concurrency: int = 1,
        parallel: bool = False,
        max_workers: int | None = None,
    ):
        self.graph = graph
        self.parallel = parallel
        self.max_workers = max_workers

        # Node-level execution context
        self._ctx: dict[str, _NodeRunner] = {
            n: _NodeRunner(node, default_concurrency) for n, node in graph.nodes.items()
        }

        # In-flight futures → (node_name, batch_id)
        self._fut2info: dict[Future, tuple[str, tuple[str, int]]] = {}

        # Buffers for assembling downstream batches
        # key: (dst_node_name, batch_id) → partial dict
        self._partial: dict[tuple[str, tuple[str, int]], dict[str, dict]] = defaultdict(
            lambda: {"pos": {}, "kw": {}}
        )

    def run(
        self,
        sources: Mapping[str, Iterable[Any]],
        *,
        batch_size: int = 1,
    ) -> Mapping[str, Any]:
        """
        Drive the workflow until all sink nodes complete.

        Args:
            sources: Mapping *source_node_name* → iterable/generator of raw data items.
            batch_size: Logical size used when chopping source iterables into batches.
        """
        with get_executor(self.parallel, self.max_workers) as executor:
            # Prime source nodes
            names = list(sources.keys())
            iterables = tuple(sources[n] for n in names)
            for ordinal, slices in batch_generator(iterables, batch_size=batch_size):
                for src_name, payload in zip(names, slices):
                    batch_id = (src_name, ordinal)
                    self._propagate_from_source(src_name, batch_id, payload, executor)

            # Drain until all futures complete
            while self._fut2info:
                self._collect_next_done(executor)

        # Gather sink outputs
        return {s: self.graph.nodes[s].results_ref for s in self.graph.sinks()}

    def _collect_next_done(self, executor: Executor) -> None:
        done, _ = wait(self._fut2info.keys(), return_when=FIRST_COMPLETED)
        for fut in done:
            node_name, batch_id = self._fut2info.pop(fut)
            try:
                result = fut.result()
            except Exception as exc:
                logger.error(f"Node {node_name} failed on batch {batch_id}: {exc!r}")
                # Retry logic
                node = self.graph.nodes[node_name]
                policy = node.retry_policy
                if node.attempt < policy.max_retries:
                    node.attempt += 1
                    backoff = policy.backoff_seconds * (node.attempt or 1)
                    logger.info(f"Retrying in {backoff:.2f}s...")
                    time.sleep(backoff)
                    retry_payload = self._retry_payloads[(node_name, batch_id)]
                    self._submit_to_node(node_name, batch_id, retry_payload, executor)
                else:
                    raise  # propagate
            else:
                self._propagate(node_name, batch_id, result, executor)

    def _propagate(
        self,
        src_name: str,
        batch_id: tuple[str, int],
        result: Any,
        executor: Executor,
    ) -> None:
        """Feed *result* down every outgoing edge."""
        for edge in self.graph.children(src_name):
            dst_batch_id = batch_id if not edge.broadcast else (edge.dst, -1)
            key = (edge.dst, dst_batch_id)
            val = edge.apply(result)

            self._partial[key]["pos"][edge.pos] = val

            # once we have *all* required inputs...
            got_pos = set(self._partial[key]["pos"].keys())
            want_pos = self.graph.input_pos(edge.dst)

            if got_pos == want_pos:
                payload = [self._partial[key]["pos"][i] for i in sorted(want_pos)]
                self._partial.pop(key)
                self._submit_to_node(edge.dst, dst_batch_id, payload, executor)

    def _submit_to_node(
        self,
        node_name: str,
        batch_id: tuple[str, int],
        payload: Iterable[Any],
        executor: Executor,
    ) -> None:
        runner = self._ctx[node_name]
        if not runner.can_accept():
            self._collect_next_done(executor)
        fut = runner.submit(payload, executor)
        self._fut2info[fut] = (node_name, batch_id)
        # remember the payload so we can retry *this* batch only
        if not hasattr(self, "_retry_payloads"):
            self._retry_payloads = {}
        self._retry_payloads[(node_name, batch_id)] = payload

    def _propagate_from_source(
        self,
        src_name: str,
        batch_id: tuple[str, int],
        payload: Iterable[Any],
        executor: Executor,
    ) -> None:
        """Initial injection from a source iterable."""
        self._propagate(src_name, batch_id, payload, executor)
        self._submit_to_node(src_name, batch_id, payload, executor)
