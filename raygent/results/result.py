# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scientific Computing Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.


from typing import Generic, TypeVar

from dataclasses import dataclass

T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    value: T | None
    """Computed values returned by [`do()`][task.Task.do]."""


@dataclass
class IndexedResult(Result[T]):
    """
    A Result with a batch index holding either a value or an error.
    """

    index: int
    """Index to specify the order of batches."""


@dataclass
class MeanResult(Result[T]):
    count: int
    """Number of values used in this mean"""

    value: T | None
    """Element-wise mean"""
