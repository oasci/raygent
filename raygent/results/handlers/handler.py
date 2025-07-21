# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scienting Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.


from typing import Any, Generic, TypeVar

from abc import ABC, abstractmethod

from loguru import logger

from raygent.savers import Saver

T = TypeVar("T")


class ResultsHandler(ABC, Generic[T]):
    """
    Abstract base class for handling results from [`Task`][task.Task].
    """

    def __init__(self, saver: Saver[T] | None = None, save_interval: int = 1) -> None:
        """
        Args:
            saver: An instance of a Saver responsible for persisting results. If None,
                no saving is performed.
            save_interval: The minimum number of results required before triggering
                a save.
        """
        self.saver: Saver[T] | None = saver
        self.save_interval: int = save_interval

    @abstractmethod
    def add_result(
        self,
        result: Any,
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
    def get(self) -> Any:
        """Get the Results"""

    def save(self) -> None:
        """
        Persists a slice of the currently collected results if a saving interval
        has been met.

        This method is typically called by a TaskRunner after each batch is added.
        If a Saver is provided and the number of collected results meets or exceeds
        the specified save_interval, a slice of results is saved using the saver,
        and the saved results are removed from the in-memory collection.
        """
        logger.warning(f"save is not implemented on {self}")

    def finalize(self) -> None:
        """Finish up any handling"""
        logger.warning(f"finalize is not implemented on {self}")
