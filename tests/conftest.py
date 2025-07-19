import os

import pytest
import ray

from raygent import enable_logging

TEST_DIR = os.path.dirname(__file__)


@pytest.fixture(scope="session", autouse=True)
def setup_tests():
    enable_logging(10)
    if not ray.is_initialized():
        ray.init(runtime_env={})


@pytest.fixture
def path_tmp():
    return os.path.join(TEST_DIR, "tmp")
