from typing import override

import pytest

from raygent import DAG, BatchMessage, Task, batch_generator


class SquareTask(Task):
    @override
    def do(self, batch: list[int]) -> list[int]:
        return [x * x for x in batch]


class PrefactorTask(Task):
    @override
    def do(self, batch: list[int], factor: int = 2) -> list[int]:
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


class _CombineTask(Task):
    """Merge two list batches into a dict so that `SumTask` can consume them."""

    @override
    def do(self, a: list[int], b: list[int]) -> dict[str, list[int]]:
        return {"a": a, "b": b}


def test_multi_node_dag_pipeline():
    """A three-stage pipeline exercising multi-input alignment & back-pressure.

    Layout::

        list1 -> SquareTask    --\
                                 CombineTask -> SumTask -> (q_out)
        list2 -> SquareTask    --/

    Expectations:
        - `SumTask` outputs are deterministic & ordered (index 0,1,…).
        - Bounded queues prevent uncontrolled growth (all set to size 2).
    """

    dag = DAG()
    sq1, source_1 = dag.add_source(SquareTask(), 1)
    sq2, source_2 = dag.add_source(SquareTask(), 1)

    comb = dag.add(_CombineTask(), inputs=(sq1, sq2))
    summed, sink_1 = dag.add_sink(SumTask(), inputs=(comb,))
    dag.run()

    list1 = [1, 2, 3, 4]
    list2 = [5, 6, 7, 8]
    batch_gen = batch_generator((list1, list2), batch_size=2)

    for index, payloads in batch_gen:
        source_1.put(BatchMessage(index=index, payload=payloads[0]))
        source_2.put(BatchMessage(index=index, payload=payloads[1]))

    expected = [[26, 40], [58, 80]]
    for idx in range(2):
        msg = sink_1.get(timeout=5)
        assert msg.index == idx
        assert expected[idx] == msg.payload
