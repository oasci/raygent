from typing import TypeVar

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

DatumType = TypeVar("DatumType")
"""
A single datum (float, NumPy array, dataframe, etc.) that can be put into
[batches][typing.BatchType]. This can be the same type as
[`BatchType`][typing.BatchType] if appropriate (e.g., NumPy array).
"""

BatchType = TypeVar("BatchType", bound=Sequence[object] | npt.NDArray[np.floating])
"""A batch container that holds many [`DatumType`][typing.DatumType]."""

OutputType = TypeVar("OutputType")
"""
Output of a [`Task.do`][task.Task.do] that accepts
[batches of `DatumType`][typing.BatchType].
"""


NumericType = TypeVar("NumericType", bound=int | float | npt.NDArray[np.floating])
