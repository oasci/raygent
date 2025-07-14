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
    value: OutputType | None = None
    error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
