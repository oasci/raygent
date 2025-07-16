import numpy as np
import numpy.typing as npt
import pytest

from raygent.results import MeanResult
from raygent.results.handlers import OnlineMeanResultsHandler
from raygent.results.result import IndexedResult


def test_mean_initial_state():
    handler = OnlineMeanResultsHandler[npt.NDArray[np.float64]]()
    assert handler.mean is None
    assert handler.total_count == 0
    with pytest.raises(ValueError):
        _ = handler.get()


def test_mean_single_batch():
    handler = OnlineMeanResultsHandler[npt.NDArray[np.float64]]()

    mean = np.array([5.0])
    count = 10
    result = IndexedResult[MeanResult[npt.NDArray[np.float64]]](
        value=MeanResult(count=10, value=mean), index=0
    )
    handler.add_result(result)
    results = handler.get()
    assert results.value is not None
    np.testing.assert_array_almost_equal(results.value, mean)
    assert results.count == count


def test_mean_multiple_batches():
    handler = OnlineMeanResultsHandler[npt.NDArray[np.float64]]()

    result = IndexedResult[MeanResult[npt.NDArray[np.float64]]](
        value=MeanResult(count=5, value=np.array([10.0])), index=0
    )
    handler.add_result(result)

    result = IndexedResult[MeanResult[npt.NDArray[np.float64]]](
        value=MeanResult(count=15, value=np.array([20.0])), index=0
    )
    handler.add_result(result)
    results = handler.get()

    # Expected mean = (5*10 + 15*20) / (5 + 15) = 350 / 20 = 17.5
    assert results.value is not None
    np.testing.assert_array_almost_equal(results.value, np.array([17.5]))
    assert results.count == 20
