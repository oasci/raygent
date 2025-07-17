from typing import Any

import itertools
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from loguru import logger

from raygent.workflow.envelope import BatchEnvelope, BatchId
from raygent.workflow.graph import WorkflowGraph


@dataclass(slots=True)
class WorkflowRunner:
    graph: WorkflowGraph
    max_cpus: int | None = None  # reserved for future resource tokens

    def run(
        self,
        sources: Mapping[str, Iterable[Any]],
        batch_size: int = 1,
    ) -> Mapping[str, Any]:
        """
        Execute the workflow synchronously (serial scheduling for now).
        `sources` maps *source node* → iterable of data batches.
        """
        # Buffers tracking partial downstream input
        partial: dict[tuple[str, BatchId], dict[str, Any]] = defaultdict(dict)
        # Track expected dst_key counts
        expected_keys = {
            n: {e.dst_key for e in self.graph.parents(n)} for n in self.graph.nodes
        }

        # Track sink results
        sink_results: dict[str, Any] = {}

        # Per‑source ordinal counters
        ctrs = {src: itertools.count() for src in self.graph.sources()}

        # Prime source envelopes
        inbox: deque[tuple[str, BatchEnvelope[Any] | None]] = deque()
        for src, iterable in sources.items():
            for ordinal, batch in zip(range(10_000_000), iterable):
                env = BatchEnvelope(batch_id=(src, ordinal), data=batch)
                inbox.append((src, env))
            # signal done with sentinel (None)
            inbox.append((src, None))

        # Main loop
        while inbox:
            src_name, envelope = inbox.popleft()
            if envelope is None:
                # Source is exhausted – mark node completed
                node = self.graph.nodes[src_name]
                node.mark_finished()
                logger.info(f"Source node {src_name} completed.")
                continue

            # Deliver envelope along each outgoing edge
            for edge in self.graph.children(src_name):
                dst = edge.dst
                batch_id = envelope.batch_id if not edge.broadcast else ("", 0)
                key = (dst, batch_id)
                contributed = edge.apply(envelope.data)
                partial[key][edge.dst_key] = contributed

                # Check if downstream batch is ready
                if partial_ready := (
                    expected_keys[dst] and partial[key].keys() >= expected_keys[dst]
                ):
                    batch_dict = partial.pop(key)
                    dst_node = self.graph.nodes[dst]
                    runner = dst_node._resolve_runner()

                    # Submit single‑batch workload (serial)
                    handler = runner.submit_tasks(
                        [batch_dict],  # one batch
                        batch_size=batch_size,
                        prebatched=True,
                        args_task=tuple(),
                        kwargs_task={},
                    )
                    result_val = handler.get()
                    dst_node.mark_finished()

                    # Enqueue result for its children
                    env_out = BatchEnvelope(
                        batch_id=(dst, batch_id[1]), data=result_val
                    )
                    for child_edge in self.graph.children(dst):
                        inbox.append((dst, env_out))

                    # collect sink outputs
                    if dst in self.graph.sinks():
                        sink_results[dst] = result_val

        return sink_results
