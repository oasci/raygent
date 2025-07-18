import pytest
import ray

from raygent.batch import batch_generator
from raygent.results import BatchMessage

from raygent.workflow import BoundedQueue, TaskActor
from raygent.task import Task


class SquareTask(Task[list[int]]):
    def do(self, batch: list[int]) -> list[int]:  # type: ignore[override]
        return [x * x for x in batch]


class PrefactorTask(Task[list[int]]):
    def do(self, batch: list[int], factor: int = 2) -> list[int]:  # type: ignore[override]  # noqa: D401,E501
        return [factor * x for x in batch]


class SumTask(Task[list[int]]):
    def do(self, batch: dict[str, list[int]]) -> list[int]:  # type: ignore[override]  # noqa: D401,E501
        return [a + b for a, b in zip(batch["a"], batch["b"])]


@pytest.mark.parametrize(
    "input_batch,expected",
    [([1, 2, 3], [1, 4, 9]), ([0, -2], [0, 4])],
)
def test_square_task_functional(input_batch: list[int], expected: list[int]):
    assert SquareTask().do(input_batch) == expected


@pytest.mark.parametrize(
    "input_batch,factor,expected",
    [([1, 2], 2, [2, 4]), ([3, 4], 3, [9, 12])],
)
def test_prefactor_task_functional(
    input_batch: list[int], factor: int, expected: list[int]
):
    assert PrefactorTask().do(input_batch, factor=factor) == expected


def test_sum_task_functional():
    batch = {"a": [1, 2, 3], "b": [10, 20, 30]}
    assert SumTask().do(batch) == [11, 22, 33]


def test_task_actor_square_pipeline():
    """Feed two aligned batches through a *single* SquareTask actor.

    Verifies:
        * Index alignment: outputs preserve the input index ordering.
        * Back‑pressure / blocking puts: uses a *small* queue to ensure we don't
          overflow (would raise if `maxsize` were ignored).
    """

    q_in: BoundedQueue = BoundedQueue(2)
    q_out: BoundedQueue = BoundedQueue(2)

    actor = TaskActor.remote(SquareTask(), 1)
    ray.get(actor.register_input.remote(q_in))
    ray.get(actor.register_output.remote(q_out))
    actor.run.remote()

    # Generate two batches of size=2 from the iterable [1, 2, 3, 4]
    batches = list(batch_generator([1, 2, 3, 4], batch_size=2))
    assert len(batches) == 2  # sanity

    # Feed input *synchronously* so we exercise the queue's maxsize=2 semantics
    for idx, (slice_,) in batches:
        q_in.put(BatchMessage(index=idx, payload=slice_))

    # Collect outputs – order must match ``idx``
    outputs: list[list[int]] = []
    for expected_idx, _ in batches:
        msg: _Msg = q_out.get(timeout=5)
        assert msg.index == expected_idx
        outputs.append(msg.payload)

    assert outputs == [[1, 4], [9, 16]]
