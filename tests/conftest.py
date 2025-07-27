# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scientific Computing Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.

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
