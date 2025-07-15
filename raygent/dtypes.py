from typing import TypeVar

from collections.abc import Sequence

DatumType = TypeVar("DatumType")
"""
A single datum (float, NumPy array, dataframe, etc.) that can be put into
[batches][typing.BatchType]. This can be the same type as
[`BatchType`][typing.BatchType] if appropriate (e.g., NumPy array).
"""

BatchType = TypeVar("BatchType", bound=Sequence[object])
"""A batch container that holds many [`DatumType`][typing.DatumType]."""

OutputType = TypeVar("OutputType")
"""
Output of a [`Task.do`][task.Task.do] that accepts
[batches of `DatumType`][typing.BatchType].
"""
