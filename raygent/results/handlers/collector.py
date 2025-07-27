# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scientific Computing Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.


from typing import Generic, TypeVar, override

from bisect import bisect_right
from collections.abc import MutableSequence
from dataclasses import dataclass

from loguru import logger

from raygent.results import IndexedResult
from raygent.results.handlers import ResultsHandler
from raygent.savers import Saver

T = TypeVar("T")


@dataclass
class ResultsBuffer(Generic[T]):
    indices: MutableSequence[int]
    results: MutableSequence[T]


class ResultsCollector(ResultsHandler[T]):
    """
    Handler that accumulates `Result[T]` instances and supports
    periodic flush/save and final aggregation.
    """

    def __init__(self, saver: Saver[T] | None = None, save_interval: int = 1) -> None:
        """
        Args:
            saver: An instance of a Saver responsible for persisting results. If None,
                no saving is performed.
            save_interval: The minimum number of results required before triggering
                a save.
        """
        self.n_results: int = 0
        self.buffer: ResultsBuffer[T] = ResultsBuffer(indices=[], results=[])
        super().__init__(saver, save_interval)

    @override
    def add_result(
        self,
        result: IndexedResult[T],
        *args: object,
        **kwargs: object,
    ) -> None:
        """
        Adds a batch of results to the handler. This method is typically called when a
        parallel task (such as a Ray task) completes execution.

        If a `batch_index` is provided, the batch is buffer until all prior batches
        have been processed to maintain the original submission order. If no
        `batch_index` is provided (i.e., when running sequentially), the batch
        results are appended directly to the final results.

        Args:
            results: The list of results produced by a batch.
        """
        if result.index in self.buffer.indices:
            raise ValueError(f"Index of {result.index} already exists")

        # Find where to insert so that indices stays sorted
        pos = bisect_right(self.buffer.indices, result.index)
        if result.value is not None:
            self.buffer.indices.insert(pos, result.index)
            self.buffer.results.insert(pos, result.value)

        self.n_results += 1
        self.save()

    def _save(self) -> None:
        if self.saver is None:
            logger.warning("ResultsHandler.saver cannot be None; skipping saving")
            return
        self.saver.save(self.buffer.results, indices=self.buffer.indices)
        self.buffer = ResultsBuffer(indices=[], results=[])
        self.n_results = 0

    @override
    def save(self) -> None:
        """
        Persists a slice of the currently collected results if a saving interval
        has been met.

        This method is typically called by a TaskRunner after each batch is added.
        If a Saver is provided and the number of collected results meets or exceeds
        the specified save_interval, a slice of results is saved using the saver,
        and the saved results are removed from the in-memory collection.

        Args:

        """
        if self.n_results >= self.save_interval:
            self._save()

    @override
    def finalize(self) -> None:
        self._save()

    @override
    def get(self) -> MutableSequence[T]:
        results: MutableSequence[T] = self.buffer.results
        return results
