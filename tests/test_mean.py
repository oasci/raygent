import numpy as np
import pytest

from raygent.results import MeanResult
from raygent.results.handlers import OnlineMeanResultsHandler


def test_mean_initial_state():
    handler = OnlineMeanResultsHandler()
    assert handler.global_mean is None
    assert handler.total_count == 0
    with pytest.raises(ValueError):
        handler.get()


def test_mean_single_batch():
    handler = OnlineMeanResultsHandler()
    partial_mean = np.array([5.0])
    count = 10
    handler.add_result(MeanResult(count=count, value=partial_mean))
    results = handler.get()
    np.testing.assert_array_almost_equal(results.value, partial_mean)
    assert results.count == count


def test_mean_multiple_batches():
    handler = OnlineMeanResultsHandler()
    # First batch: mean = [10.0], count = 5.
    handler.add_result(MeanResult(count=5, value=np.array([10.0])))
    # Second batch: mean = [20.0], count = 15.
    handler.add_result(MeanResult(count=15, value=np.array([20.0])))
    results = handler.get()
    # Expected mean = (5*10 + 15*20) / (5 + 15) = 350 / 20 = 17.5
    np.testing.assert_array_almost_equal(results.value, np.array([17.5]))
    assert results.count == 20
