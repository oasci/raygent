import os

import pytest
import ray

from raygent import enable_logging

TEST_DIR = os.path.dirname(__file__)


@pytest.fixture(scope="session", autouse=True)
def setup_tests():
    enable_logging(10)
    ray.init(runtime_env={}, num_cpus=16, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def path_tmp():
    return os.path.join(TEST_DIR, "tmp")
