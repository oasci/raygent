from typing import Generic, TypeVar

# ParamSpec does not have default= (PEP 696) until Python 3.13
# This import can be replaced once we stop supporting <3.13
from typing_extensions import ParamSpec

from collections.abc import Iterable, Sequence

from raygent.results import Result
from raygent.savers import Saver

OutputType = TypeVar("OutputType")

P = ParamSpec("P", default=())  # pyright: ignore[reportGeneralTypeIssues]


class ResultAccumulator(Generic[OutputType, P]):
    """
    Abstract handler that accumulates Result[OutputType] instances and supports periodic
    flush/save and final aggregation.
    """

    def add_chunk(
        self,
        results: Iterable[Result[OutputType]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """
        Adds a chunk of results to the handler. This method is typically called when a
        parallel task (such as a Ray task) completes execution.

        If a `chunk_index` is provided, the chunk is buffered until all prior chunks
        have been processed to maintain the original submission order. If no
        `chunk_index` is provided (i.e., when running sequentially), the chunk
        results are appended directly to the final results.

        Args:
            chunk_results: The list of results produced by a chunk.
            chunk_index: The index of the chunk. This value is used to
                maintain the correct order.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).
        """
        raise NotImplementedError

    def periodic_save(self, saver: Saver, save_interval: int) -> None:
        """
        Persists a slice of the currently collected results if a saving interval
        has been met.

        This method is typically called by a TaskManager after each chunk is added.
        If a Saver is provided and the number of collected results meets or exceeds
        the specified save_interval, a slice of results is saved using the saver,
        and the saved results are removed from the in-memory collection.

        Args:
            saver: An instance of a Saver responsible for persisting results. If None,
                no saving is performed.
            save_interval: The minimum number of results required before triggering
                a save.
        """
        return

    def finalize(self, saver: Saver) -> None:
        raise NotImplementedError

    def get_results(self) -> Sequence[Result[OutputType]]:
        raise NotImplementedError
