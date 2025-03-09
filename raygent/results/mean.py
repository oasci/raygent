from typing import Any

import numpy as np
import numpy.typing as npt

from raygent.results import BaseResultHandler
from raygent.savers import Saver


class OnlineMeanResultHandler(BaseResultHandler):
    r"""
    `OnlineMeanResultHandler` provides a numerically stable, online (incremental)
    algorithm to compute the arithmetic mean of large, streaming, or distributed
    datasets represented as NumPy arrays. In many real-world applications—such a
    distributed computing or real-time data processing—data is processed
    in chunks, with each chunk yielding a partial mean and its corresponding count.
    This class merges these partial results into a global mean without needing to
    store all the raw data, thus avoiding issues such as numerical overflow and
    precision loss.

    Suppose the overall dataset is divided into k chunks.
    For each chunk $i$ (where $1 \leq i \leq k$), let:

      - $m_i$ be the partial mean computed over $n_i$ data points.
      - $M_i$ be the global mean computed after processing $i$ chunks.
      - $N_i = n_1 + n_2 + ... + n_i$ be the cumulative count after $i$ chunks.

    The arithmetic mean of all data points is defined as:

    $$
    M_\text{total} = \frac{n_1 m_1 + n_2 m_2 + \ldots + n_k m_k}{n_1 + n_2 + \ldots + n_k}
    $$

    Rather than computing $M_{total}$ from scratch after processing all data,
    the class uses an iterative update rule. When merging a new partial result
    `(m_partial, n_partial)` with the current global mean `M_old` (with count
    `n_old`), the updated mean is given by:

    $$
    M_\text{new} = M_\text{old} + \left( m_\text{partial} - M_\text{old} \right) \cdot
        \frac{n_\text{partial}}{n_\text{old} + n_\text{partial}}
    $$

    This update is mathematically equivalent to the weighted average:

    $$
    M_\text{new} = \frac{n_\text{old} M_\text{old} + n_\text{partial} m_\text{partial}}{n_\text{old} + n_\text{partial}}
    $$

    but is rearranged to enhance numerical stability. By focusing on the
    difference `(m_partial - M_old)` and scaling it by the relative weight
    `n_partial / (n_old + n_partial)`, the algorithm minimizes the round-off
    errors that can occur when summing large numbers or when processing many
    chunks sequentially.
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
        chunk_results: (
            list[tuple[npt.NDArray[np.float64], int]]
            | tuple[npt.NDArray[np.float64], int]
        ),
        chunk_index: int | None = None,
        *args: tuple[Any],
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Processes one or more chunks of partial results to update the global mean.

        Args:
            chunk_results: Either a single tuple (partial_mean, count) or a list of
                such tuples. Each partial result must be a tuple consisting of:
                    - A NumPy array representing the partial mean of a data chunk.
                    - An integer representing the count of observations in that chunk.
            chunk_index: An optional index identifier for the chunk (for interface consistency, not used in calculations).
        """
        if isinstance(chunk_results, tuple):
            chunk_results = [chunk_results]

        for partial_mean, count in chunk_results:
            if self.global_mean is None:
                self.global_mean = np.array(partial_mean, dtype=np.float64)
                self.total_count = count
            else:
                new_total = self.total_count + count
                self.global_mean = self.global_mean + (
                    partial_mean - self.global_mean
                ) * (count / new_total)
                self.total_count = new_total

    def periodic_save_if_needed(self, saver: Saver | None, save_interval: int) -> None:
        """
        Persists the current global mean at periodic intervals if a saver is provided.

        This method is useful in long-running or distributed applications where saving
        intermediate results is necessary for fault tolerance. It checks if the total
        number of observations processed meets or exceeds the `save_interval` and,
        if so, invokes the save method of the provided saver.

        Args:
            saver: An instance of a Saver, which has a save method to persist data,
                or None.
            save_interval: The threshold for the total number of observations to
                trigger a save.
        """
        if saver and self.total_count >= save_interval and self.global_mean is not None:
            saver.save({"mean": self.global_mean, "n": self.total_count})

    def finalize(self, saver: Saver | None) -> None:
        """
        Finalizes the accumulation process and persists the final global mean
        if a saver is provided.

        This method should be called when no more data is expected. It ensures
        that the final computed mean and the total observation count are saved.

        Args:
            saver: An instance of a Saver to save the final result, or None.
        """
        if saver and self.global_mean is not None:
            saver.save({"mean": self.global_mean, "n": self.total_count})

    def get_results(self) -> dict[str, npt.NDArray[np.float64] | int]:
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
        if self.global_mean is None or self.total_count == 0:
            raise ValueError("No data has been processed.")
        return {"mean": self.global_mean, "n": self.total_count}
