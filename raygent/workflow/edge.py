from typing import Any, Callable, Generic, TypeVar, get_type_hints, overload

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

    def type_check(
        self,
        upstream_out_type: type[Any],
        dst_accepts_dict: bool,
    ) -> None:
        """
        Raises TypeError if selector/transform annotations conflict with
        declared upstream / downstream types.
        """
        from typing import Any as _Any  # avoid collision

        def ret(obj: Callable[..., Any]) -> Any:
            return get_type_hints(obj).get("return", _Any)

        trans_out = ret(self.transform) if self.transform else upstream_out_type

        if not dst_accepts_dict and trans_out is not _Any:
            raise TypeError(
                f"Edge {self.src}->{self.dst} produces {trans_out!r}, but downstream node does not accept dict batches."
            )
