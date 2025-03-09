from typing import Any

from abc import ABC, abstractmethod

from raygent.savers import Saver


class BaseResultHandler(ABC):
    """
    Abstract base for any "result handler" that accumulates results
    from processed chunks.
    """

    @abstractmethod
    def add_chunk(
        self,
        chunk_results: list[Any],
        chunk_index: int | None = None,
        *args: tuple[Any],
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Called when a new chunk of results is ready. Implement
        custom accumulation logic here (e.g. append to list, or
        do a running mean update, etc.).
        """
        ...

    def periodic_save_if_needed(
        self, saver: Saver | None, save_interval: int
    ) -> None:
        """
        Called periodically by the TaskManager to let the handler
        decide if/when it needs to save partial results.
        """
        ...

    @abstractmethod
    def finalize(self, saver: Saver | None) -> None:
        """
        Called one final time after all chunks have been processed,
        so any leftover data can be persisted or cleaned up.
        """
        ...

    @abstractmethod
    def get_results(self) -> list[Any] | dict[str, Any]:
        """
        Returns the "collected" results. For a Welford aggregator,
        this might return just mean/variance. For a list aggregator,
        it might return the entire list of results.
        """
        ...
