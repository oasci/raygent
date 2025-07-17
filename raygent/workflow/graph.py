from types import MappingProxyType

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass

from raygent.workflow import WorkflowEdge, WorkflowNode


@dataclass(slots=True)
class WorkflowGraph:
    """
    Pure, validated DAG: nodes + edges + typing constraints.
    """

    nodes: Mapping[str, WorkflowNode]
    edges: Sequence[WorkflowEdge]

    def __post_init__(self) -> None:
        # Freeze mapping so external callers can’t mutate after validation
        self.nodes = MappingProxyType(dict(self.nodes))

        self._validate_unique_names()
        self._validate_edges_exist()
        self._validate_unique_dst_keys()
        self._validate_acyclic()
        self._validate_edge_types()

    @classmethod
    def from_iterables(
        cls,
        nodes: Iterable[WorkflowNode],
        edges: Iterable[WorkflowEdge],
    ) -> "WorkflowGraph":
        return cls({n.name: n for n in nodes}, list(edges))

    def parents(self, node: str) -> Sequence[WorkflowEdge]:
        return [e for e in self.edges if e.dst == node]

    def children(self, node: str) -> Sequence[WorkflowEdge]:
        return [e for e in self.edges if e.src == node]

    def sources(self) -> Sequence[str]:
        return [n for n in self.nodes if not any(e.dst == n for e in self.edges)]

    def sinks(self) -> Sequence[str]:
        return [n for n in self.nodes if not any(e.src == n for e in self.edges)]

    def _validate_unique_names(self) -> None:
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Duplicate node names detected.")

    def _validate_edges_exist(self) -> None:
        node_set = set(self.nodes)
        for e in self.edges:
            if e.src not in node_set or e.dst not in node_set:
                raise ValueError(f"Edge {e} references unknown node(s).")

    def _validate_unique_dst_keys(self) -> None:
        per_node: MutableMapping[str, set[str]] = {}
        for e in self.edges:
            if e.dst not in per_node:
                per_node[e.dst] = set()
            if e.dst_key in per_node[e.dst]:
                raise ValueError(
                    f"Duplicate dst_key {e.dst_key!r} into node {e.dst!r}."
                )
            per_node[e.dst].add(e.dst_key)

    def _validate_acyclic(self) -> None:
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(node: str) -> None:
            if node in stack:
                raise ValueError("Cycle detected in workflow graph.")
            if node in visited:
                return
            stack.add(node)
            for child_edge in self.children(node):
                dfs(child_edge.dst)
            stack.remove(node)
            visited.add(node)

        for n in self.nodes:
            dfs(n)

    def _validate_edge_types(self) -> None:
        for edge in self.edges:
            src_node = self.nodes[edge.src]
            dst_node = self.nodes[edge.dst]

            # --- infer upstream output type ---------------------------- #
            tm = src_node._resolve_runner()
            upstream_out_type: type[Any] = getattr(
                tm,
                "output_type",
                Any,  # fallback; you can refine TaskRunner
            )

            # --- confirm downstream accepts dict[str, Any] ------------- #
            tm_down = dst_node._resolve_runner()
            downstream_input_type: type[Any] = getattr(tm_down, "input_type", Mapping)
            downstream_is_dict = (
                downstream_input_type in _DICT_OR_MAPPING
                or get_origin(downstream_input_type) in _DICT_OR_MAPPING
            )

            edge.type_check(upstream_out_type, downstream_is_dict)
