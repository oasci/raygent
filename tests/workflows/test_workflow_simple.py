from typing import override

import pytest
import ray

from raygent.batch import batch_generator
from raygent.results import BatchMessage
from raygent.task import Task
from raygent.workflow import DAG, BoundedQueue, TaskActor


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


def test_task_actor_square_pipeline():
    """Feed two aligned batches through a *single* SquareTask actor.

    Verifies:
        - Index alignment: outputs preserve the input index ordering.
        - Back-pressure / blocking puts: uses a *small* queue to ensure we don't
          overflow (would raise if `maxsize` were ignored).
    """

    q_in: BoundedQueue[list[int]] = BoundedQueue(maxsize=2)
    q_out: BoundedQueue[list[int]] = BoundedQueue(maxsize=2)

    actor = TaskActor.remote(SquareTask(), 1)
    _ = ray.get(actor.register_input.remote(q_in))
    _ = ray.get(actor.register_output.remote(q_out))
    _ = actor.run.remote()

    # Generate two batches of size=2 from the iterable [1, 2, 3, 4]
    batches = list(batch_generator([1, 2, 3, 4], batch_size=2))
    assert len(batches) == 2

    # Feed input *synchronously* so we exercise the queue's maxsize=2 semantics
    for idx, (slice_,) in batches:
        q_in.put(BatchMessage(index=idx, payload=slice_))

    # Collect outputs – order must match ``idx``
    outputs: list[list[int]] = []
    for expected_idx, _ in batches:
        msg = q_out.get(timeout=5)
        assert msg.index == expected_idx
        outputs.append(msg.payload)

    assert outputs == [[1, 4], [9, 16]]


class _CombineTask(Task[dict[str, list[int]]]):
    """Merge two list batches into a dict so that `SumTask` can consume them."""

    @override
    def do(self, a: list[int], b: list[int]) -> dict[str, list[int]]:
        return {"a": a, "b": b}


def test_multi_node_dag_pipeline():
    """A three-stage pipeline exercising multi-input alignment & back-pressure.

    Layout::

        list1 -> SquareTask    --\
                                  CombineTask -> SumTask -> (q_out)
        list2 -> PrefactorTask --/

    Expectations:
        - `SumTask` outputs are deterministic & ordered (index 0,1,…).
        - Bounded queues prevent uncontrolled growth (all set to size 2).
    """

    q_in_a = BoundedQueue(2)
    q_in_b = BoundedQueue(2)

    q_mid_a = BoundedQueue(2)
    q_mid_b = BoundedQueue(2)

    q_combined = BoundedQueue(2)
    q_out = BoundedQueue(2)

    square_actor = TaskActor.remote(SquareTask(), 1)
    ray.get(square_actor.register_input.remote(q_in_a))
    ray.get(square_actor.register_output.remote(q_mid_a))
    square_actor.run.remote()

    prefactor_actor = TaskActor.remote(PrefactorTask(), 1)
    ray.get(prefactor_actor.register_input.remote(q_in_b))
    ray.get(prefactor_actor.register_output.remote(q_mid_b))
    prefactor_actor.run.remote()

    combiner_actor = TaskActor.remote(_CombineTask(), 2)
    ray.get(combiner_actor.register_input.remote(q_mid_a))
    ray.get(combiner_actor.register_input.remote(q_mid_b))
    ray.get(combiner_actor.register_output.remote(q_combined))
    combiner_actor.run.remote()

    sum_actor = TaskActor.remote(SumTask(), 1)
    ray.get(sum_actor.register_input.remote(q_combined))
    ray.get(sum_actor.register_output.remote(q_out))
    sum_actor.run.remote()

    list1 = [1, 2, 3, 4]
    list2 = [5, 6, 7, 8]
    batches1 = list(batch_generator(list1, batch_size=2))
    batches2 = list(batch_generator(list2, batch_size=2))
    assert len(batches1) == len(batches2) == 2

    for (idx_a, (slice_a,)), (idx_b, (slice_b,)) in zip(batches1, batches2):
        # Ensure indices match across both inputs
        assert idx_a == idx_b
        q_in_a.put(BatchMessage(index=idx_a, payload=slice_a))
        q_in_b.put(BatchMessage(index=idx_b, payload=slice_b))

    expected = [[11, 16], [23, 32]]
    actual: list[list[int]] = []
    for idx in range(2):
        msg = q_out.get(timeout=5)
        assert msg.index == idx
        actual.append(msg.payload)

    assert actual == expected


def test_multi_node_dag_pipeline_add():
    """A three-stage pipeline exercising multi-input alignment & back-pressure.

    Layout::

        list1 -> SquareTask    --\
                                  CombineTask -> SumTask -> (q_out)
        list2 -> PrefactorTask --/

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
    batches1 = list(batch_generator(list1, batch_size=2))
    batches2 = list(batch_generator(list2, batch_size=2))
    assert len(batches1) == len(batches2) == 2

    for (idx_a, (slice_a,)), (idx_b, (slice_b,)) in zip(batches1, batches2):
        assert idx_a == idx_b
        source_1.put(BatchMessage(index=idx_a, payload=slice_a))
        source_2.put(BatchMessage(index=idx_b, payload=slice_b))

    expected = [[11, 16], [23, 32]]
    actual: list[list[int]] = []
    for idx in range(2):
        msg = sink_1.get(timeout=5)
        assert msg.index == idx
        actual.append(msg.payload)

    assert actual == expected
