from typing import TypeVarTuple, Unpack

from collections.abc import Generator, Iterable
from itertools import islice

Ts = TypeVarTuple("Ts")


def batch_generator(
    *iterables: Unpack[tuple[Iterable[*Ts]]],
    batch_size: int = 1,
    prebatched: bool = False,
    max_batches: int = 1_000_000_000,
) -> Generator[tuple[int, tuple[list[*Ts], ...]], None, None]:
    """
    Yields (batch_index, (slice1, slice2, …)), where each slice
    is up to `batch_size` items from the corresponding iterable.

    Call as:
        batch_generator((it1, it2, …), batch_size=…)
    """

    assert batch_size > 0, "batch_size must be positive"

    # allow a single tuple/list of iterables to be passed directly:
    if (
        not prebatched
        and len(iterables) == 1
        and isinstance(iterables[0], (tuple, list))
        and all(isinstance(it, Iterable) for it in iterables[0])
    ):
        iterables = tuple(iterables[0])

    if prebatched:
        # if the inputs are already "batches", just zip them together
        for idx, batch in enumerate(zip(*iterables)):
            yield idx, batch  # each element of `batch` is already a tuple
        return

    # otherwise: pull up to batch_size from each iterator, in lock‑step
    its = tuple(iter(it) for it in iterables)
    for idx in range(max_batches):
        parts: list[list[*Ts]] = []
        for it in its:
            chunk = list(islice(it, batch_size))
            if not chunk:
                # stop as soon as any iterator is exhausted
                return
            parts.append(chunk)
        yield idx, tuple(parts)
