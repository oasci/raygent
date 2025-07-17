from typing import Callable, Generic, TypeVar, overload

from dataclasses import dataclass

SourceT = TypeVar("SourceT")
"""Type of the source node (i.e., upstream) output."""

TailT = TypeVar("TailT")
"""Type of the data the sink node (i.e., downstream) expects from the upstream node."""


@dataclass(slots=True)
class WorkflowEdge(Generic[SourceT, TailT]):
    """
    Directed, typed connection between two WorkflowNodes.
    """

    src: str
    """Name of source node."""

    dst: str
    """Name of destination node."""

    dst_key: str
    """Name of data this edge provides to `dst`."""

    transform: Callable[[SourceT], TailT] | None = None
    """A function that transforms data from the source node to the tail node."""

    broadcast: bool = False
    """If this edge broadcasts a constant value."""

    @overload
    def apply(
        self: "WorkflowEdge[SourceT, SourceT]", upstream_value: SourceT
    ) -> SourceT: ...

    @overload
    def apply(self, upstream_value: SourceT) -> TailT: ...

    def apply(self, upstream_value: SourceT) -> TailT | SourceT:
        v = upstream_value
        if self.transform is not None:
            v = self.transform(v)
        return v
