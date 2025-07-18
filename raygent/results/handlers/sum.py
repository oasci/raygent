from typing import TypeVar, override

from raygent.results import Result
from raygent.results.handlers import ResultsHandler
from raygent.savers import Saver

T = TypeVar("T")


class SumResultsHandler(ResultsHandler[T]):
    """Sum all results."""

    def __init__(self, saver: Saver[T] | None = None, save_interval: int = 1) -> None:
        """
        Args:
            saver: An instance of a Saver responsible for persisting results. If None,
                no saving is performed.
            save_interval: The minimum number of results required before triggering
                a save.
        """

        self.sum: T | None = None
        """
        The current global sum of all processed results.
        """
        super().__init__(saver, save_interval)

    @override
    def add_result(
        self,
        result: Result[T],
        *args: object,
        **kwargs: object,
    ) -> None:
        """
        Processes one or more batches of partial results to update the global mean.
        """
        if result.value is None:
            return
        if self.sum is None:
            self.sum = result.value
        else:
            self.sum += result.value

    @override
    def get(self) -> T:
        """
        Retrieves the final computed global mean along with the total number of
        observations.

        Returns:
            A dictionary with the following keys:

                -   `"mean"`: A NumPy array representing the computed global mean.
                -   `"n"`: An integer representing the total number of observations
                    processed.

        Raises:
            ValueError: If no data has been processed (i.e., `global_mean` is None or
                `total_count` is zero).
        """
        if self.sum is None:
            raise ValueError("No data has been processed.")
        return self.sum
