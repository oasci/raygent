from typing import Any

from raygent.results import BaseResultHandler
from raygent.savers import Saver


class ListResultHandler(BaseResultHandler):
    """
    A result handler that preserves the submission (chunk) order when collecting
    results in parallel. This handler uses an internal dictionary to store chunks
    keyed by their chunk index, and flushes completed chunks in ascending order (0, 1, 2, ...)
    into a final list. This ensures that even if chunks complete out of order, the final
    results are returned in the correct submission order.
    """

    def __init__(self) -> None:
        self._collected: list[Any] = []
        """
        A list containing the final, ordered results after flushing.
        """

        self._buffered: dict[int, list[Any]] = {}
        """
        A temporary storage for chunks that are not yet flushed,
        keyed by their chunk index.
        """

        self._next_index: int = 0
        """
        The index of the next expected chunk to flush from the buffered dictionary.
        """

    def add_chunk(
        self,
        chunk_results: list[Any],
        chunk_index: int | None = None,
        *args: tuple[Any],
        **kwargs: dict[str, Any],
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

    def periodic_save_if_needed(self, saver: Saver | None, save_interval: int) -> None:
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
        if not saver or len(self._collected) < save_interval:
            return

        # Save a slice of results:
        to_save = self._collected[:save_interval]
        saver.save(to_save)

        # Remove those results from the “collected” in-memory store
        self._collected = self._collected[save_interval:]

    def finalize(self, saver: Saver | None) -> None:
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
        # If there's anything left in the buffer (unlikely if the
        # manager waits for all tasks), flush it now:
        while self._next_index in self._buffered:
            next_chunk = self._buffered.pop(self._next_index)
            self._collected.extend(next_chunk)
            self._next_index += 1

        # If we still have some unsaved results, now is a good time to save them
        if saver and self._collected:
            saver.save(self._collected)

    def get_results(self) -> list[Any]:
        """
        Retrieves all results collected by the handler in their original submission order.

        Returns:
            A list containing all the results in order.
        """
        return self._collected
