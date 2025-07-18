from typing import override

from collections.abc import Iterable

import pytest

from raygent import Task, TaskRunner
from raygent.results.handlers.collector import ResultsCollector
from raygent.workflow.edge import WorkflowEdge
from raygent.workflow.graph import WorkflowGraph
from raygent.workflow.node import RetryPolicy, WorkflowNode
from raygent.workflow.runners import WorkflowRunner


class SquareTask(Task[list[int]]):
    @override
    def do(self, batch: list[int]) -> list[int]:
        return [x * x for x in batch]


class PrefactorTask(Task[list[int]]):
    @override
    def do(self, batch: list[int], factor: int = 2) -> list[int]:
        return [factor * x for x in batch]


class SumTask(Task[list[int]]):
    @override
    def do(self, batch: dict[str, list[int]]) -> list[int]:
        # zip-sum corresponding positions (assumes equal length lists)
        return [a + b for a, b in zip(batch["a"], batch["b"])]


class MultiplyTask(Task[list[int]]):
    """Multiply incoming list by a constant broadcast edge 'factor'."""

    @override
    def do(self, batch: list[int], factor: float = 2.0) -> list[float]:
        return [x * factor for x in batch]


@pytest.fixture()
def range_batches() -> dict[str, Iterable[list[int]]]:
    # Produce 3 one-element batches for two sources
    return {
        "src_a": [[1], [2], [3]],
        "src_b": [[10], [20], [30]],
    }


def _node(name: str, task_cls, **kwargs) -> WorkflowNode:
    return WorkflowNode(
        name=name,
        runner=TaskRunner(task_cls, ResultsCollector, in_parallel=False),
        **kwargs,
    )


def test_linear_pipeline():
    """src -> square -> double -> sink."""
    node_src = _node("src", SquareTask)
    node_double = _node("double", PrefactorTask)
    node_sink = _node("sink", PrefactorTask)

    edges = [
        WorkflowEdge("src", "double", pos=0),
        WorkflowEdge("double", "sink", pos=0),
    ]

    graph = WorkflowGraph.from_iterables([node_src, node_double, node_sink], edges)
    runner = WorkflowRunner(graph, parallel=False, default_concurrency=2)
    outs = runner.run({"src": [1, 2]}, batch_size=1)

    assert outs["sink"] == [[4], [16]]  # (square -> double) twice


def test_fan_in_selector_transform(range_batches):
    """Two sources -> transform on one edge -> join-sum -> sink."""
    node_a = _node("src_a", SquareTask)  # 1², 2², 3²
    node_b = _node("src_b", SquareTask)  # 10², 20², 30²
    node_join = _node("join", SumTask)

    # Edge from B doubles values after squaring (transform demonstration)
    edge_ab = WorkflowEdge(
        "src_b",
        "join",
        "b",
        transform=lambda lst: [2 * x for x in lst],
    )

    graph = WorkflowGraph.from_iterables(
        [node_a, node_b, node_join],
        [
            WorkflowEdge("src_a", "join", "a"),
            edge_ab,
        ],
    )

    runner = WorkflowRunner(graph)
    outs = runner.run(range_batches, batch_size=1)

    # Expected: [1+200, 4+800, 9+1800] = [201, 804, 1809]
    assert outs["join"] == [[201], [804], [1809]]


def test_broadcast_constant(range_batches):
    """Broadcast a single factor (constant edge) to all data batches."""
    cfg_node = _node("config", PrefactorTask)
    data_node = _node("data_src", SquareTask)
    mult_node = _node("multiply", MultiplyTask)

    edges = [
        WorkflowEdge("data_src", "multiply", "batch"),
        WorkflowEdge(
            "config", "multiply", "factor", broadcast=True, transform=lambda _: 3
        ),
    ]

    graph = WorkflowGraph.from_iterables(
        [cfg_node, data_node, mult_node],
        edges,
    )

    runner = WorkflowRunner(graph)
    outs = runner.run(
        {
            "config": [["dummy"]],  # single element -> broadcast
            "data_src": [[2], [4]],
        },
        batch_size=1,
    )

    # square then *3
    assert outs["multiply"] == [[12], [48]]


def test_sparse_batch_ids():
    """One source missing ordinal=1 - downstream should process 0 & 2 only."""

    src_a_iter = [[1], [3]]
    src_b_iter = [[10], [30]]

    # drop second element from src_a to make it sparse
    src_a_iter.pop(1)  # now ordinals 0 only

    node_a = _node("src_a", SquareTask)
    node_b = _node("src_b", SquareTask)
    node_join = _node("join", SumTask)

    graph = WorkflowGraph.from_iterables(
        [node_a, node_b, node_join],
        [
            WorkflowEdge("src_a", "join", "a"),
            WorkflowEdge("src_b", "join", "b"),
        ],
    )

    runner = WorkflowRunner(graph)
    outs = runner.run(
        {"src_a": src_a_iter, "src_b": src_b_iter},
        batch_size=1,
    )

    # Only ordinal 0 had both parents
    assert outs["join"] == [[1 + 100]]  # 1² + 10²


def test_retry_policy(range_batches):
    """Artificially raise on first attempt to test RetryPolicy."""

    class FlakyTask(Task[list[int]]):
        def __init__(self):
            self._called = False

        def do(self, batch: list[int]) -> list[int]:
            if not self._called:
                self._called = True
                raise RuntimeError("boom")
            return batch

    node_src = _node(
        "src",
        FlakyTask,
        retry_policy=RetryPolicy(max_retries=1, backoff_seconds=0.01),
    )
    graph = WorkflowGraph.from_iterables([node_src], [])
    runner = WorkflowRunner(graph)
    outs = runner.run({"src": [[5]]})
    # should succeed after retry
    assert outs["src"] == [[5]]
