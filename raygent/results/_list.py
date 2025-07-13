from typing import Generic, TypeVar

# ParamSpec does not have default= (PEP 696) until Python 3.13
# This import can be replaced once we stop supporting <3.13
from typing_extensions import ParamSpec

from collections.abc import Iterable

from raygent.results import Result, ResultAccumulator
from raygent.savers import Saver

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")

P = ParamSpec("P", default=())  # pyright: ignore[reportGeneralTypeIssues]


class ListResults(ResultAccumulator[OutputType, P]):
    """
    Collects Result[OutputType] in original order, supports buffering out-of-order
    and periodic save of completed windows.
    """

    def __init__(self):
        self._buffered: dict[int, Iterable[Result[OutputType]]] = {}
        self._ordered: Iterable[Result[OutputType]] = []
        self._next_index: int = 0

    def add_chunk(
        self,
        results: Iterable[Result[OutputType]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        # buffer by index, flush contiguous
        for res in results:
            self._buffered[res.index] = self._buffered.get(res.index, []) + [res]
        while self._next_index in self._buffered:
            chunk = self._buffered.pop(self._next_index)
            self._ordered.extend(chunk)
            self._next_index += 1

    def finalize(self, saver: Saver) -> None:
        """
        Finalizes the result collection process by flushing any remaining buffered
        chunks and persisting any unsaved results if a saver is provided.

        This method should be called after all chunks have been submitted and
        completed. It ensures that any remaining results in the buffer are appended
        to the final results, and if a Saver is available, it saves the unsaved results.

        Args:
            saver: An instance of a Saver used to persist the final results. If None,
                no saving is performed.
        """
        while self._next_index in self._buffered:
            chunk = self._buffered.pop(self._next_index)
            self._ordered.extend(chunk)
            self._next_index += 1
        # optionally save if saver provided
        if saver:
            saver.save(self._ordered)

    def get_results(self) -> Iterable[Result[OutputType]]:
        return self._ordered
