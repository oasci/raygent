# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scienting Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.


import os
import shutil

import numpy as np
import pytest

from raygent.savers import ZarrSaver

try:
    import zarr

    has_zarr = True
except ImportError:
    has_zarr = False


@pytest.fixture
def test_data() -> list[int]:
    """Simple list of integer data for testing."""
    return [1, 2, 3, 4, 5]


@pytest.mark.skipif(not has_zarr, reason="Zarr is not installed.")
class TestZarrSaver:
    @pytest.fixture
    def zarr_temp_path(self, path_tmp: str) -> str:
        """
        Returns a temporary path for testing Zarr.
        The final path looks like: /tmp/pytest-of-<user>/pytest-<test>/test.zarr
        """
        return os.path.join(path_tmp, "test.zarr")

    def test_append(self, zarr_temp_path, test_data):
        shutil.rmtree(zarr_temp_path, ignore_errors=True)
        saver = ZarrSaver(
            file_path=zarr_temp_path, dataset_name="dataset", approach="append"
        )
        saver.save(test_data)

        # Reopen to verify data
        store = zarr.storage.LocalStore(zarr_temp_path, read_only=False)
        z = zarr.open_array(store, path="dataset")
        assert np.array_equal(z[:], np.array(test_data))

        # Append more data
        saver.save([6, 7])
        z = zarr.open_array(store, path="dataset")
        assert np.array_equal(z[:], np.array([1, 2, 3, 4, 5, 6, 7]))

    def test_overwrite(self, zarr_temp_path, test_data):
        shutil.rmtree(zarr_temp_path, ignore_errors=True)
        append_saver = ZarrSaver(
            file_path=zarr_temp_path, dataset_name="dataset", approach="overwrite"
        )
        append_saver.save(test_data)

        # Now overwrite
        overwrite_saver = ZarrSaver(
            file_path=zarr_temp_path, dataset_name="dataset", approach="overwrite"
        )
        new_data = [100, 200, 300]
        overwrite_saver.save(new_data)

        # Reopen to verify data was overwritten
        store = zarr.storage.LocalStore(zarr_temp_path, read_only=False)
        z = zarr.open_array(store, path="dataset")
        assert np.array_equal(z[:], np.array(new_data))

    def test_update(self, zarr_temp_path, test_data):
        shutil.rmtree(zarr_temp_path, ignore_errors=True)
        saver = ZarrSaver(
            file_path=zarr_temp_path, dataset_name="dataset", approach="overwrite"
        )
        saver.save(test_data)  # [1, 2, 3, 4, 5]

        # Now update the second and third elements to 99, 99
        update_saver = ZarrSaver(
            file_path=zarr_temp_path, dataset_name="dataset", approach="update"
        )
        update_saver.save([99, 99], indices=[1, 2])  # place starting at index=1

        store = zarr.storage.LocalStore(zarr_temp_path, read_only=False)
        z = zarr.open_array(store, path="dataset")
        assert np.array_equal(z[:], np.array([1, 99, 99, 4, 5]))

    def test_update_nonexistent_dataset(self, zarr_temp_path, test_data):
        """
        Ensure that update on a non-existent dataset raises ValueError.
        """
        saver = ZarrSaver(
            file_path=zarr_temp_path, dataset_name="unknown_dataset", approach="update"
        )
        with pytest.raises(FileNotFoundError):
            saver.save(test_data)


@pytest.mark.skipif(not has_zarr, reason="Zarr is not installed.")
def test_zarr_import_error(monkeypatch, test_data, path_tmp):
    """
    If you want to test that a missing Zarr installation raises ImportError.
    We can monkey-patch has_zarr to False and ensure .save(...) raises ImportError.
    """
    from raygent.savers import ZarrSaver

    class FakeZarrSaver(ZarrSaver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    # Force has_zarr = False
    monkeypatch.setattr("raygent.savers._zarr.has_zarr", False)

    saver = FakeZarrSaver(file_path="dummy.zarr", dataset_name="dummy")
    with pytest.raises(ImportError):
        saver.save(test_data)
