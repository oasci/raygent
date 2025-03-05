import os

import numpy as np
import pytest

from raygent.savers import NumpySaver

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False


@pytest.fixture
def test_data() -> list[int]:
    """Simple list of integer data for testing."""
    return [1, 2, 3, 4, 5]


class TestNumpySaver:
    @pytest.fixture
    def numpy_temp_path(self, path_tmp: str) -> str:
        return os.path.join(path_tmp, "test.npy")

    def test_overwrite(self, numpy_temp_path, test_data):
        if os.path.exists(numpy_temp_path):
            os.remove(numpy_temp_path)
        saver = NumpySaver(file_path=numpy_temp_path, approach="overwrite")
        saver.save(test_data)

        # Load and check
        loaded = np.load(numpy_temp_path)
        assert np.array_equal(loaded, np.array(test_data))

        # Overwrite with new data
        new_data = [100, 200]
        saver.save(new_data)
        loaded = np.load(numpy_temp_path)
        assert np.array_equal(loaded, np.array(new_data))

    def test_append(self, numpy_temp_path, test_data):
        if os.path.exists(numpy_temp_path):
            os.remove(numpy_temp_path)
        saver = NumpySaver(file_path=numpy_temp_path, approach="append")
        saver.save(test_data)  # [1, 2, 3, 4, 5]
        loaded = np.load(numpy_temp_path)
        assert np.array_equal(loaded, np.array(test_data))

        # Append more data
        saver.save([6, 7])
        loaded = np.load(numpy_temp_path)
        assert np.array_equal(loaded, np.array([1, 2, 3, 4, 5, 6, 7]))

    def test_bad_approach(self, numpy_temp_path, test_data):
        with pytest.raises(ValueError):
            saver = NumpySaver(file_path=numpy_temp_path, approach="unknown")
            saver.save(test_data)
