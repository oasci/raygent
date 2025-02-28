import os

import pytest
from raygent import enable_logging

TEST_DIR = os.path.dirname(__file__)


@pytest.fixture(scope="session", autouse=True)
def turn_on_logging():
    enable_logging(10)


@pytest.fixture
def path_tmp():
    return os.path.join(TEST_DIR, "tmp")
