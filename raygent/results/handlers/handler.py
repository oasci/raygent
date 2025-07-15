from typing import Generic

from abc import ABC, abstractmethod
from collections.abc import Sequence

from loguru import logger

from raygent.dtypes import OutputType
from raygent.results import Result
from raygent.savers import Saver


class ResultsHandler(ABC, Generic[OutputType]):
    """
    Abstract base class for handling results from [`Task`][task.Task].
    """

    def __init__(
        self, saver: Saver[OutputType] | None = None, save_interval: int = 1
    ) -> None:
        """
        Args:
            saver: An instance of a Saver responsible for persisting results. If None,
                no saving is performed.
            save_interval: The minimum number of results required before triggering
                a save.
        """
        self.saver: Saver[OutputType] | None = saver
        self.save_interval: int = save_interval

    @abstractmethod
    def add_result(
        self,
        result: Result[OutputType],
        *args: object,
        **kwargs: object,
    ) -> None:
        """
        Handles a [`Result`][results.result.Result] to the handler. This method is
        typically called when a [`Task`][task.Task] completes execution.

        Args:
            result: A [`Result`][results.result.Result] produced by a batch.
        """

    @abstractmethod
    def get(self) -> Sequence[OutputType] | object:
        """Get the Results"""

    def save(self) -> None:
        """
        Persists a slice of the currently collected results if a saving interval
        has been met.

        This method is typically called by a TaskManager after each batch is added.
        If a Saver is provided and the number of collected results meets or exceeds
        the specified save_interval, a slice of results is saved using the saver,
        and the saved results are removed from the in-memory collection.
        """
        logger.warning(f"save is not implemented on {self}")

    def finalize(self) -> None:
        """Finish up any handling"""
        logger.warning(f"finalize is not implemented on {self}")
