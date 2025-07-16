from typing import override

from raygent.dtypes import NumericType
from raygent.results import MeanResult
from raygent.results.handlers import ResultsHandler
from raygent.results.result import IndexedResult
from raygent.savers import Saver


class OnlineMeanResultsHandler(ResultsHandler[NumericType]):
    r"""
    `OnlineMeanResultsHandler` provides a numerically stable, online (incremental)
    algorithm to compute the arithmetic mean of large, streaming, or distributed
    datasets represented as NumPy arrays. In many real-world applications—such a
    distributed computing or real-time data processing—data is processed
    in batches, with each batch yielding a partial mean and its corresponding count.
    This class merges these partial results into a global mean without needing to
    store all the raw data, thus avoiding issues such as numerical overflow and
    precision loss.

    Suppose the overall dataset is divided into k batches.
    For each batch $i$ (where $1 \leq i \leq k$), let:

      - $m_i$ be the partial mean computed over $n_i$ data points.
      - $M_i$ be the global mean computed after processing $i$ batches.
      - $N_i = n_1 + n_2 + ... + n_i$ be the cumulative count after $i$ batches.

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
    batches sequentially.
    """

    def __init__(
        self, saver: Saver[NumericType] | None = None, save_interval: int = 1
    ) -> None:
        """
        Args:
            saver: An instance of a Saver responsible for persisting results. If None,
                no saving is performed.
            save_interval: The minimum number of results required before triggering
                a save.
        """

        self.mean: NumericType | None = None
        """
        The current global mean of all processed observations.
        """

        self.total_count: int = 0
        """
        The total number of observations processed.
        """
        super().__init__(saver, save_interval)

    @override
    def add_result(
        self,
        result: IndexedResult[MeanResult[NumericType]],
        *args: object,
        **kwargs: object,
    ) -> None:
        """
        Processes one or more batches of partial results to update the global mean.
        """
        if result.value is None:
            return
        else:
            mean_result = result.value

        if mean_result.value is None:
            return

        if self.mean is None:
            self.mean = mean_result.value
            self.total_count = mean_result.count
            return

        new_total = self.total_count + mean_result.count
        self.mean = self.mean + (mean_result.value - self.mean) * (
            mean_result.count / new_total
        )
        self.total_count = new_total

    @override
    def get(self) -> MeanResult[NumericType]:
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
        if self.mean is None or self.total_count == 0:
            raise ValueError("No data has been processed.")
        return MeanResult[NumericType](value=self.mean, count=self.total_count)
