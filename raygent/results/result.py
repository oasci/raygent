from typing import Any, Generic, TypeVar

from dataclasses import dataclass, field

OutputType = TypeVar("OutputType")


@dataclass
class Result(Generic[OutputType]):
    """
    A generic result wrapper holding either a value or an error,
    plus metadata for ordering, retries, and diagnostics.
    """

    index: int
    """Index to specify the order of chunks."""

    value: OutputType | None = None
    """Computed values returned by [`do()`][task.Task.do]."""

    error: Exception | None = None
    """Errors that are captured during [`do()`][task.Task.do]."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Future-proofing metadata."""
