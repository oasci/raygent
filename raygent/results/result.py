from typing import Protocol

from dataclasses import dataclass

from raygent.dtypes import OutputType


class Result(Protocol[OutputType]):
    value: OutputType | None
    """Computed values returned by [`do()`][task.Task.do]."""


@dataclass
class IndexedResult(Result[OutputType]):
    """
    A Result with a batch index holding either a value or an error.
    """

    index: int
    """Index to specify the order of batches."""

    value: OutputType | None = None
    """Computed values returned by [`do()`][task.Task.do]."""


@dataclass
class MeanResult(Result[OutputType]):
    count: int
    """Number of values used in this mean"""

    value: OutputType | None
    """Element-wise mean"""
