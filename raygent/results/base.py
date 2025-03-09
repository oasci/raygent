from typing import Any

from abc import ABC, abstractmethod

from raygent.savers import Saver


class BaseResultHandler(ABC):
    """
    Abstract base class for result handlers that accumulate results from
    processed chunks.

    This class defines the interface for custom result handling strategies,
    such as collecting results in a list, computing running statistics
    (e.g., mean/variance), or any other aggregation approach. Implementations must
    override the abstract methods to provide concrete behavior.
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
        Incorporates a new chunk of results into the handler.

        This method is called when a chunk of results is ready, such as after a
        parallel task completes. The method should implement the logic to update the
        handler's state with the new results. If the results need to be ordered
        (e.g., based on chunk_index), the implementation should ensure that the order
        is maintained.

        Args:
            chunk_results: A list containing the results of the current chunk.
            chunk_index: The index of the chunk, used to maintain order. If None,
                the handler may assume that the results are already in the correct
                order.
            *args: Additional positional arguments that may be used by specific
                implementations.
            **kwargs: Additional keyword arguments that may be used by
                specific implementations.
        """
        ...

    def periodic_save_if_needed(self, saver: Saver | None, save_interval: int) -> None:
        """
        Optionally saves partial results based on a defined save interval.

        This method is called periodically (e.g., after processing each chunk) by the
        TaskManager to determine if the current accumulated results should be
        persisted to a storage medium. Implementations may override this method to
        incorporate saving logic, or leave it as a no-op
        if saving is handled differently.

        Args:
            saver: An instance of a Saver responsible for persisting results. If None,
                no saving operation is performed.
            save_interval: The number of results that should trigger a save operation.
        """
        ...

    @abstractmethod
    def finalize(self, saver: Saver | None) -> None:
        """
        Finalizes the result collection process by flushing any remaining data and
        saving it if required.

        This method is called once after all chunks have been processed. It provides
        an opportunity to persist any remaining results or perform any necessary
        cleanup. The implementation should ensure that the final state of the
        accumulated results is complete and ready for retrieval.

        Args:
            saver: An instance of a Saver used to persist the final accumulated results.
                If None, no saving operation is performed.
        """
        ...

    @abstractmethod
    def get_results(self) -> list[Any] | dict[str, Any]:
        """
        Retrieves the accumulated results.

        Depending on the implementation, this method may return a list, a dictionary,
        or another data structure representing the collected results. For instance,
        a list aggregator may return all individual results in order, while a
        statistical aggregator might return computed metrics like mean and variance.

        Returns:
            The accumulated results in their appropriate format.
        """
        ...
