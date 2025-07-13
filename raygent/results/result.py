from typing import Any, Generic, TypeVar

# ParamSpec does not have default= (PEP 696) until Python 3.13
# This import can be replaced once we stop supporting <3.13
from typing_extensions import ParamSpec

from dataclasses import dataclass, field

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")

P = ParamSpec("P", default=())  # pyright: ignore[reportGeneralTypeIssues]


@dataclass
class Result(Generic[OutputType]):
    """
    A generic result wrapper holding either a value or an error,
    plus metadata for ordering, retries, and diagnostics.
    """

    index: int
    value: OutputType | None = None
    error: Exception | None = None
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
