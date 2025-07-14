from typing import Generic, TypeVar

from bisect import bisect_right

from loguru import logger

from raygent.results import Result
from raygent.savers import Saver

OutputType = TypeVar("OutputType")


class ResultHandler(Generic[OutputType]):
    """
    Abstract handler that accumulates `Result[OutputType]` instances and supports periodic
    flush/save and final aggregation.
    """

    def __init__(self, saver: Saver | None = None, save_interval: int = 1) -> None:
        """
        Args:
            saver: An instance of a Saver responsible for persisting results. If None,
                no saving is performed.
            save_interval: The minimum number of results required before triggering
                a save.
        """
        self.n_results: int = 0
        self.buffer: dict[str, list[int | OutputType]] = {"indices": [], "results": []}
        self.saver: Saver | None = saver
        self.save_interval: int = save_interval

    def add_result(
        self,
        result: Result[OutputType],
    ) -> None:
        """
        Adds a chunk of results to the handler. This method is typically called when a
        parallel task (such as a Ray task) completes execution.

        If a `chunk_index` is provided, the chunk is buffer until all prior chunks
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
        if result.index in self.buffer.keys():
            raise ValueError(f"Index of {result.index} already exists")

        # Find where to insert so that indices stays sorted
        pos = bisect_right(self.buffer["indices"], result.index)

        # Splice into both lists at the same position
        self.buffer["indices"].insert(pos, result.index)
        self.buffer["results"].insert(pos, result.value)

        self.n_results += 1
        self.save()

    def _save(self) -> None:
        if self.saver is None:
            logger.warning("ResultHandler.saver cannot be None; skipping saving")
            return
        self.saver.save(self.buffer["results"], indices=self.buffer["indices"])
        self.buffer = {"indices": [], "results": []}
        self.n_results = 0

    def save(self) -> None:
        """
        Persists a slice of the currently collected results if a saving interval
        has been met.

        This method is typically called by a TaskManager after each chunk is added.
        If a Saver is provided and the number of collected results meets or exceeds
        the specified save_interval, a slice of results is saved using the saver,
        and the saved results are removed from the in-memory collection.

        Args:

        """
        if self.n_results >= self.save_interval:
            self._save()

    def finalize(self) -> None:
        self._save()

    def get(self) -> list[OutputType]:
        results: list[OutputType] = self.buffer["results"]
        return results
