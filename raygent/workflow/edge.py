from typing import Any, Callable, Generic, ParamSpec, TypeVar, get_type_hints

from collections.abc import Mapping
from dataclasses import dataclass, field

UpOutT = TypeVar("UpOutT")  # upstream OutputType
SelT = TypeVar("SelT")  # after selector
DownInT = TypeVar("DownInT")  # final contribution to dst batch

Selector = Callable[[UpOutT], SelT]
Transform = Callable[[SelT], DownInT]

P = ParamSpec("P")  # not used yet but reserved for future hooks


@dataclass(slots=True)
class WorkflowEdge(Generic[UpOutT, SelT, DownInT]):
    """
    Directed, typed connection between two WorkflowNodes.
    """

    src: str
    dst: str
    dst_key: str
    selector: Selector | None = None
    transform: Transform | None = None
    broadcast: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def type_check(
        self,
        upstream_out_type: type[Any],
        downstream_field_type: type[Any],
    ) -> None:
        """
        Perform a static‑ish type compatibility check using annotations.
        Raises TypeError on mismatch.
        """

        # Helper to extract the -> return annotation
        def ret_ann(callable_obj: Callable[..., Any]) -> Any:
            return get_type_hints(callable_obj).get("return", Any)

        sel_out = upstream_out_type
        if self.selector:
            sel_out = ret_ann(self.selector)

        trans_out = sel_out
        if self.transform:
            trans_out = ret_ann(self.transform)

        # Very pragmatic compatibility rule: exact match or Any or subclass.
        if (
            trans_out is not downstream_field_type
            and downstream_field_type is not Any
            and not issubclass(trans_out, downstream_field_type)
        ):
            raise TypeError(
                f"Edge {self.src}->{self.dst!r} ({self.dst_key}) produces"
                f" {trans_out}, but downstream expects {downstream_field_type}."
            )

    def apply(self, upstream_value: UpOutT) -> DownInT:
        v: Any = upstream_value
        if self.selector is not None:
            v = self.selector(v)
        if self.transform is not None:
            v = self.transform(v)  # type: ignore[arg-type]
        return v  # type: ignore[return-value]
