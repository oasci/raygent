from typing import Generic, TypeAlias, TypeVar

from dataclasses import dataclass

T = TypeVar("T")

BatchId: TypeAlias = tuple[str, int]


@dataclass(slots=True)
class BatchEnvelope(Generic[T]):
    """Internal wrapper used by WorkflowRunner for routing & sync."""

    batch_id: BatchId
    data: T
