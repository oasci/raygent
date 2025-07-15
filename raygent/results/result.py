from typing import Generic

from dataclasses import dataclass, field

from raygent.dtypes import OutputType


@dataclass
class Result(Generic[OutputType]):
    """
    A generic result wrapper holding either a value or an error,
    plus metadata for ordering, retries, and diagnostics.
    """

    index: int
    """Index to specify the order of batches."""

    value: OutputType | None = None
    """Computed values returned by [`do()`][task.Task.do]."""

    error: Exception | None = None
    """Errors that are captured during [`do()`][task.Task.do]."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Future-proofing metadata."""
