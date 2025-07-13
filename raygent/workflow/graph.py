import collections
import graphlib

from loguru import logger

from raygent.workflow import WorkflowTaskNode



class WorkflowGraph:
    """
    Manages the Directed Acyclic Graph (DAG) structure of the workflow.
    Responsible for adding tasks and defining dependencies, and topological sorting.
    """

    def __init__(self, workflow_name: str):
        self.workflow_name: str = workflow_name
        self._nodes: dict[str, WorkflowTaskNode] = {}
        self._dependencies: dict[str, list[str]] = collections.defaultdict(list)

    def add_node(
        self, node: WorkflowTaskNode, depends_on: str | list[str] | None = None
    ):
        """Adds a task node to the graph and establishes dependencies."""
        if node.name in self._nodes:
            raise ValueError(
                f"Task with name '{node.name}' already exists in workflow '{self.workflow_name}'."
            )

        self._nodes[node.name] = node

        if depends_on:
            if isinstance(depends_on, str):
                self._dependencies[node.name].append(depends_on)
            else:
                self._dependencies[node.name].extend(depends_on)

        logger.info(
            f"Workflow '{self.workflow_name}': Added task '{node.name}' (depends on: {self._dependencies[node.name]})."
        )

    def get_node(self, task_name: str) -> WorkflowTaskNode:
        """Retrieves a task node by its name."""
        if task_name not in self._nodes:
            raise KeyError(
                f"Task '{task_name}' not found in workflow '{self.workflow_name}'."
            )
        return self._nodes[task_name]

    def get_all_nodes(self) -> dict[str, WorkflowTaskNode]:
        """Returns all task nodes in the graph."""
        return self._nodes

    def get_dependencies(self, task_name: str) -> list[str]:
        """Returns the upstream dependencies for a given task."""
        return self._dependencies[task_name]

    def topological_sort(self) -> list[str]:
        """
        Performs a topological sort of the workflow tasks to determine execution order.

        Raises:
            ValueError: If a circular dependency is detected.
        Returns:
            A list of task names in a valid execution order.
        """
        tsorter = graphlib.TopologicalSorter()
        for task_name in self._nodes.keys():
            tsorter.add(task_name, *self._dependencies[task_name])

        try:
            order = list(tsorter.static_order())
            logger.debug(f"Workflow '{self.workflow_name}': Topological order: {order}")
            return order
        except graphlib.CycleError as e:
            raise ValueError(
                f"Circular dependency detected in workflow '{self.workflow_name}': {e}"
            )

