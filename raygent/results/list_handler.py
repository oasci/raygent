from typing import Any

from raygent.results import BaseResultHandler
from raygent.savers import Saver


class ListResultHandler(BaseResultHandler):
    """
    A result handler that preserves the submission (chunk) order
    when collecting results in parallel.  Internally uses a dictionary
    keyed by chunk_index, and flushes completed chunks in ascending
    order (0,1,2,...) to a final list.
    """

    def __init__(self) -> None:
        # Stores the final, ordered results after we've flushed them.
        self._collected: list[Any] = []

        # Temporarily holds chunks keyed by chunk_index until we can
        # place them in order:
        self._buffered: dict[int, list[Any]] = {}

        # The next chunk index we expect to flush:
        self._next_index: int = 0

    def add_chunk(
        self,
        chunk_results: list[Any],
        chunk_index: int | None = None,
        *args: tuple[Any],
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Called when a chunk of results has finished (e.g. after
        a parallel Ray task completes). We store them in a dictionary
        so that, if chunks finish out of order, we can still reconstruct
        the final results in their original submission order.
        """
        if chunk_index is None:
            # If no index was provided (i.e. we're running sequentially),
            # just pretend everything is in order:
            self._collected.extend(chunk_results)
            return

        # 1) Buffer the results under their chunk_index
        self._buffered[chunk_index] = chunk_results

        # 2) Attempt to flush any contiguous “next_index, next_index+1, etc.”
        while self._next_index in self._buffered:
            # Move that chunk out of the buffer:
            next_chunk = self._buffered.pop(self._next_index)
            # Append to the final collected list:
            self._collected.extend(next_chunk)
            self._next_index += 1

    def periodic_save_if_needed(
        self, saver: Saver | None, save_interval: int
    ) -> None:
        """
        Called by the TaskManager after each chunk is added to let
        the handler decide if it's time to persist partial results.
        """
        if not saver or len(self._collected) < save_interval:
            return

        # Save a slice of results:
        to_save = self._collected[:save_interval]
        saver.save(to_save)

        # Remove those results from the “collected” in-memory store
        self._collected = self._collected[save_interval:]

    def finalize(self, saver: Saver | None) -> None:
        """
        Called after all chunks have been submitted and completed.
        We can flush anything that's left over in memory.
        """
        # If there's anything left in the buffer (unlikely if the
        # manager waits for all tasks), flush it now:
        while self._next_index in self._buffered:
            next_chunk = self._buffered.pop(self._next_index)
            self._collected.extend(next_chunk)
            self._next_index += 1

        # If we still have some unsaved results, now is a good time to save them
        if saver and self._collected:
            saver.save(self._collected)
            # Typically you'd either:
            # 1) clear self._collected entirely, or
            # 2) keep it so get_results() can still return everything
            # We'll keep it in memory so get_results() can return the full data.

    def get_results(self) -> list[Any]:
        """
        Returns all results in their original, correct submission order.
        """
        return self._collected
