import numpy as np
import pytest
from raygent.results import OnlineMeanResultHandler


class DummySaver:
    """A simple saver to record saved data."""

    def __init__(self):
        self.saved_data = []

    def save(self, data):
        self.saved_data.append(data)


def test_mean_initial_state():
    handler = OnlineMeanResultHandler()
    assert handler.global_mean is None
    assert handler.total_count == 0
    with pytest.raises(ValueError):
        handler.get_results()


def test_mean_single_chunk():
    handler = OnlineMeanResultHandler()
    partial_mean = np.array([5.0])
    count = 10
    handler.add_chunk((partial_mean, count))
    results = handler.get_results()
    np.testing.assert_array_almost_equal(results["mean"], partial_mean)
    assert results["n"] == count


def test_mean_multiple_chunks():
    handler = OnlineMeanResultHandler()
    # First chunk: mean = [10.0], count = 5.
    handler.add_chunk((np.array([10.0]), 5))
    # Second chunk: mean = [20.0], count = 15.
    handler.add_chunk((np.array([20.0]), 15))
    results = handler.get_results()
    # Expected mean = (5*10 + 15*20) / (5 + 15) = 350 / 20 = 17.5
    np.testing.assert_array_almost_equal(results["mean"], np.array([17.5]))
    assert results["n"] == 20


def test_mean_periodic_save_and_finalize(DummySaver):
    handler = OnlineMeanResultHandler()
    saver = DummySaver()
    handler.add_chunk((np.array([3.0]), 3))
    # Trigger periodic saving with a save_interval lower than total_count.
    handler.periodic_save_if_needed(saver, save_interval=2)
    # Finalize should also trigger a save.
    handler.finalize(saver)
    # Expect two saves.
    assert len(saver.saved_data) == 2
    last_save = saver.saved_data[-1]
    np.testing.assert_array_almost_equal(last_save["mean"], np.array([3.0]))
    assert last_save["n"] == 3
