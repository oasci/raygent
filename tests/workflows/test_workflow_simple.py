# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scienting Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.

from typing import override

import pytest

from raygent import Task
from raygent.workflow import DAG


class SquareTask(Task):
    @override
    def do(self, batch: list[int]) -> list[int]:
        return [x * x for x in batch]


class PrefactorTask(Task):
    @override
    def do(self, batch: list[int], factor: int = 1) -> list[int]:
        return [factor * x for x in batch]


class SumTask(Task):
    @override
    def do(self, batch: dict[str, list[int]]) -> list[int]:
        return [a + b for a, b in zip(batch["a"], batch["b"])]


@pytest.mark.parametrize(
    "input_batch,expected",
    [([1, 2, 3], [1, 4, 9]), ([0, -2], [0, 4])],
)
def test_square_task_functional(input_batch: list[int], expected: list[int]) -> None:
    assert SquareTask().do(input_batch) == expected


@pytest.mark.parametrize(
    "input_batch,factor,expected",
    [([1, 2], 2, [2, 4]), ([3, 4], 3, [9, 12])],
)
def test_prefactor_task_functional(
    input_batch: list[int], factor: int, expected: list[int]
) -> None:
    assert PrefactorTask().do(input_batch, factor=factor) == expected


def test_sum_task_functional() -> None:
    batch = {"a": [1, 2, 3], "b": [10, 20, 30]}
    assert SumTask().do(batch) == [11, 22, 33]


class CombineTask(Task):
    """Merge two list batches into a dict so that `SumTask` can consume them."""

    @override
    def do(self, a: list[int], b: list[int]) -> dict[str, list[int]]:
        return {"a": a, "b": b}


def test_multi_node_dag_pipeline():
    """A three-stage pipeline exercising multi-input alignment & back-pressure.

    Layout:

        source_1 -> (PrefactorTask) --> (SquareTask)  --\
                                                     (CombineTask) -> (SumTask) -> sink_1
        source_2 -----> (SquareTask) ------------------/
                                 |
                                 ----> sink_2
    """

    dag = DAG()

    # Add queues we can send data into the DAG
    source_n1, source_1 = dag.add_source()
    source_n2, source_2 = dag.add_source()

    # Add fully-connected nodes to process our workflow
    ## Top
    prefactor_n = dag.add(PrefactorTask(), inputs=source_n1, task_kwargs={"factor": 2})
    square_n1 = dag.add(SquareTask(), inputs=prefactor_n)
    ## Bottom
    square_n2 = dag.add(SquareTask(), inputs=source_n2)
    ## Combined
    comb = dag.add(CombineTask(), inputs=(square_n1, square_n2))
    summed = dag.add(SumTask(), inputs=comb)

    # Attach sinks to get data out of our workflow
    sink_1 = dag.add_sink(summed)
    sink_2 = dag.add_sink(square_n2)

    data1 = [1, 2, 3, 4]
    data2 = [5, 6, 7, 8]

    expected_sink1 = [[29, 52], [85, 128]]
    expected_sink2 = [[25, 36], [49, 64]]

    sources = (source_1, source_2)
    sinks = (sink_1, sink_2)

    n_messages = 0
    dag.start()
    for sink_idx, msg in dag.stream(
        data1,
        data2,
        source_queues=sources,
        sink_queues=sinks,
        batch_size=2,
        max_inflight=4,
    ):
        print(
            f"from sink #{sink_idx}  ->  batch_idx={msg.index}, payload={msg.payload}"
        )
        n_messages += 1
        if sink_idx == 0:
            assert expected_sink1[msg.index] == msg.payload
        if sink_idx == 1:
            assert expected_sink2[msg.index] == msg.payload
    dag.stop()
    assert n_messages == 4
