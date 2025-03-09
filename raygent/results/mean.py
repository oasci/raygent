from typing import Any

import numpy as np
import numpy.typing as npt

from raygent.results import BaseResultHandler
from raygent.savers import Saver


class OnlineMeanResultHandler(BaseResultHandler):
    """
    Computes an online, element-wise mean for NumPy arrays using a numerically stable,
    parallel algorithm designed to avoid overflow and minimize numerical error.

    In distributed or streaming data settings, each task computes a partial mean over
    a subset of data and returns a tuple: (partial_mean, count). Rather than directly summing
    all values—which can lead to numerical overflow or loss of precision—the handler combines
    these partial results into a global mean.

    The merging uses a weighted update formula based on the counts of the observations:

        M_{new} = M_{old} + (m_{partial} - M_{old}) * (n_{partial} / (n_{old} + n_{partial}))

    where:
      - \( M_{old} \) is the current global mean,
      - \( m_{partial} \) is the mean computed for the new data chunk,
      - \( n_{old} \) is the number of observations included in the current global mean,
      - \( n_{partial} \) is the number of observations in the new chunk.

    This formula is derived from the weighted average:

        M_{new} = \frac{n_{old} \cdot M_{old} + n_{partial} \cdot m_{partial}}{n_{old} + n_{partial}},

    which can be rearranged to the above update form. This rearrangement is more stable when
    processing large data values or when combining many chunks sequentially.
    """

    def __init__(self) -> None:
        """
        The handler starts with no accumulated data. The global mean (`global_mean`) is initially
        set to None, and it will be defined by the first partial result received. The total number
        of observations (`total_count`) is initialized to zero.
        """

        self.global_mean: npt.NDArray[np.float64] | None = None
        """
        The current global mean of all processed observations.
        """

        self.total_count: int = 0
        """
        The total number of observations processed.
        """

    def add_chunk(
        self,
        chunk_results: list[tuple[npt.NDArray[np.float64], int]] | tuple[npt.NDArray[np.float64], int],
        chunk_index: int | None = None,
        *args: tuple[Any],
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Processes one or more chunks of partial results to update the global mean.

        Each partial result must be a tuple consisting of:
            - A NumPy array representing the partial mean of a data chunk.
            - An integer representing the count of observations in that chunk.

        The update rule for merging a new partial result \((m_{partial}, n_{partial})\) with the
        current global result \((M_{old}, n_{old})\) is as follows:

            \( M_{new} = M_{old} + \left(m_{partial} - M_{old}\right) \times \frac{n_{partial}}{n_{old} + n_{partial}} \)
            \( n_{new} = n_{old} + n_{partial} \)

        This formula is mathematically equivalent to computing the weighted average:

            \( M_{new} = \frac{n_{old} \times M_{old} + n_{partial} \times m_{partial}}{n_{old} + n_{partial}} \)

        but it provides better numerical stability when combining a large number of values.

        Args:
            chunk_results: Either a single tuple (partial_mean, count) or a list of such tuples.
            chunk_index: An optional index identifier for the chunk (for interface consistency, not used in calculations).
        """
        # If a single tuple is provided, wrap it in a list.
        if isinstance(chunk_results, tuple):
            chunk_results = [chunk_results]

        for partial_mean, count in chunk_results:
            if self.global_mean is None:
                # Initialize global_mean with the first partial mean.
                self.global_mean = np.array(partial_mean, dtype=np.float64)
                self.total_count = count
            else:
                # Merge the new partial mean with the current global mean.
                new_total = self.total_count + count
                # Compute the weighted difference and update the global mean.
                self.global_mean = self.global_mean + (partial_mean - self.global_mean) * (count / new_total)
                self.total_count = new_total

    def periodic_save_if_needed(self, saver: Saver | None, save_interval: int) -> None:
        """
        Persists the current global mean at periodic intervals if a saver is provided.

        This method is useful in long-running or distributed applications where saving intermediate
        results is necessary for fault tolerance. It checks if the total number of observations processed
        meets or exceeds the `save_interval` and, if so, invokes the save method of the provided saver.

        Args:
            saver: An instance of a Saver, which has a save method to persist data, or None.
            save_interval: The threshold for the total number of observations to trigger a save.
        """
        if saver and self.total_count >= save_interval and self.global_mean is not None:
            saver.save({"mean": self.global_mean, "n": self.total_count})

    def finalize(self, saver: Saver | None) -> None:
        """
        Finalizes the accumulation process and persists the final global mean if a saver is provided.

        This method should be called when no more data is expected. It ensures that the final
        computed mean and the total observation count are saved.

        Args:
            saver: An instance of a Saver to save the final result, or None.
        """
        if saver and self.global_mean is not None:
            saver.save({"mean": self.global_mean, "n": self.total_count})

    def get_results(self) -> dict[str, npt.NDArray[np.float64] | int]:
        """
        Retrieves the final computed global mean along with the total number of observations.

        Returns:
            A dictionary with the following keys:
                - "mean": A NumPy array representing the computed global mean.
                - "n": An integer representing the total number of observations processed.

        Raises:
            ValueError: If no data has been processed (i.e., `global_mean` is None or `total_count` is zero).
        """
        if self.global_mean is None or self.total_count == 0:
            raise ValueError("No data has been processed.")
        return {"mean": self.global_mean, "n": self.total_count}
