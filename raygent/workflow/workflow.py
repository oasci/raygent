from typing import Any

from collections.abc import Callable, Iterable

try:
    import ray

    has_ray = True
except ImportError:
    has_ray = False

from loguru import logger

from raygent import TaskManager
from raygent.savers import Saver
from raygent.workflow import WorkflowTaskNode, WorkflowGraph
from raygent.workflow.utils import _chain_iterables

TaskManagerFactory = Callable[[], TaskManager[Any, Any]]
TaskManagerOrFactory = TaskManager[Any, Any] | TaskManagerFactory




class Workflow:
    """
    Orchestrates a series of interconnected tasks, forming a Directed Acyclic Graph (DAG).
    """

    def __init__(self, name: str, ray_init_kwargs: dict[str, Any] | None = None):
        self.name: str = name
        self._graph: WorkflowGraph = WorkflowGraph(name)
        self._initialize_ray(ray_init_kwargs)

    def _initialize_ray(self, ray_init_kwargs: dict[str, Any] | None):
        """Initializes Ray if not already initialized and configured to do so."""
        if has_ray and ray_init_kwargs is not None:
            if not ray.is_initialized():
                logger.info(
                    f"Workflow '{self.name}': Initializing Ray with {ray_init_kwargs or 'defaults'}."
                )
                ray.init(**ray_init_kwargs)
            else:
                logger.info(f"Workflow '{self.name}': Ray already initialized.")
        elif has_ray and not ray.is_initialized() and ray_init_kwargs is None:
            logger.info(f"Workflow '{self.name}': Initializing Ray with defaults.")
            ray.init()

    def add_task(
        self,
        task_name: str,
        task_manager_or_factory: TaskManagerOrFactory,
        depends_on: str | list[str] | None = None,
        output_to_disk: bool = False,
        output_saver: Saver | None = None,
        output_chunk_size: int | None = None,
        kwargs_task: dict[str, Any] | None = None,  # Task-specific kwargs_task
        kwargs_remote: dict[str, Any] | None = None,  # Task-specific kwargs_remote
    ) -> None:
        """
        Adds a task (represented by a TaskManager) to the workflow.
        """
        if output_to_disk and output_saver is None:
            raise ValueError(
                "If output_to_disk is True, an output_saver must be provided."
            )

        node = WorkflowTaskNode(
            name=task_name,
            manager_or_factory=task_manager_or_factory,
            output_to_disk=output_to_disk,
            output_saver=output_saver,
            output_chunk_size=output_chunk_size,
            task_specific_kwargs_task=kwargs_task,
            task_specific_kwargs_remote=kwargs_remote,
        )
        self._graph.add_node(node, depends_on)

    def _resolve_input_data(
        self,
        task_node: WorkflowTaskNode,
        initial_data: Any | None,
        task_outputs: "dict[str, list[Any] | ray.ObjectRef | str]",
    ) -> Iterable[Any]:
        """
        Resolves and combines the input data for a given task node.
        This method has the single responsibility of providing the correct input.
        """
        input_sources = []

        # Collect inputs from completed upstream dependencies
        for upstream_task_name in self._graph.get_dependencies(task_node.name):
            upstream_output = task_outputs.get(upstream_task_name)
            if upstream_output is None:
                # This indicates a logical error in topological sort or dependency management
                raise RuntimeError(
                    f"Workflow '{self.name}': Upstream task '{upstream_task_name}' output not found for '{task_node.name}'."
                )
            input_sources.append(upstream_output)

        # If no dependencies, and initial_data is provided, it's the input
        if (
            not self._graph.get_dependencies(task_node.name)
            and initial_data is not None
        ):
            input_sources.append(initial_data)

        if not input_sources:
            logger.warning(
                f"Workflow '{self.name}': Task '{task_node.name}' has no input data."
            )
            return []  # Return an empty iterable

        # Combine multiple inputs (e.g., from multiple upstream dependencies)
        return _chain_iterables(*input_sources)

    def _execute_task(
        self,
        task_node: WorkflowTaskNode,
        input_data: Iterable[Any],
        global_kwargs: dict[str, Any],
    ) -> "list[Any] | ray.ObjectRef | str":
        """
        Executes a single task node using its TaskManager.
        This method has the single responsibility of running the task and getting its raw output.
        """
        logger.info(f"Workflow '{self.name}': Executing task '{task_node.name}'.")
        task_node.update_status("running")

        try:
            task_manager_instance = task_node.get_task_manager_instance()

            # Merge global and task-specific kwargs, with task-specific overriding globals
            merged_kwargs_task = {
                **global_kwargs["kwargs_task"],
                **task_node.task_specific_kwargs_task,
            }
            merged_kwargs_remote = {
                **global_kwargs["kwargs_remote"],
                **task_node.task_specific_kwargs_remote,
            }

            # Handle output saving for this task
            current_saver: Saver | None = None
            current_save_interval = global_kwargs["save_interval"]
            if task_node.output_to_disk:
                current_saver = task_node.output_saver
                if task_node.output_chunk_size is not None:
                    current_save_interval = task_node.output_chunk_size

            task_manager_instance.submit_tasks(
                items=input_data,
                saver=current_saver,
                chunk_size=global_kwargs["chunk_size"],
                at_once=global_kwargs["at_once"],
                save_interval=current_save_interval,
                kwargs_task=merged_kwargs_task,
                kwargs_remote=merged_kwargs_remote,
            )

            # Get results from TaskManager (materialized list or a path/identifier from saver)
            task_raw_results = task_manager_instance.get_results()

            # Determine how to store/pass the results
            if task_node.output_to_disk:
                result_for_workflow = f"Disk_Saved_Results_for_{task_node.name}"
                logger.info(
                    f"Workflow '{self.name}': Task '{task_node.name}' results saved to disk."
                )
            elif has_ray and task_manager_instance.use_ray:
                # If Ray is used, put results into Ray object store for efficient passing
                result_for_workflow = ray.put(task_raw_results)
                logger.debug(
                    f"Workflow '{self.name}': Task '{task_node.name}' results placed in Ray object store."
                )
            else:
                # For sequential execution, keep results in Python memory
                result_for_workflow = task_raw_results
                logger.debug(
                    f"Workflow '{self.name}': Task '{task_node.name}' results kept in memory."
                )

            task_node.update_status("completed")
            task_node.results = result_for_workflow  # Store reference/list on the node
            return result_for_workflow

        except Exception as e:
            task_node.update_status("failed", str(e))
            logger.error(f"Workflow '{self.name}': Task '{task_node.name}' failed: {e}")
            raise RuntimeError(
                f"Workflow '{self.name}': Task '{task_node.name}' failed."
            ) from e

    def run(
        self,
        initial_data: Any | None = None,
        chunk_size: int = 100,
        at_once: bool = False,
        save_interval: int = 100,
        kwargs_task: dict[str, Any] | None = None,
        kwargs_remote: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Executes the workflow based on the defined tasks and dependencies.
        """
        all_nodes = self._graph.get_all_nodes()
        if not all_nodes:
            raise ValueError(f"Workflow '{self.name}' has no tasks defined.")

        execution_order = self._graph.topological_sort()
        logger.info(f"Workflow '{self.name}': Starting execution.")

        global_kwargs = {
            "chunk_size": chunk_size,
            "at_once": at_once,
            "save_interval": save_interval,
            "kwargs_task": kwargs_task or {},
            "kwargs_remote": kwargs_remote or {},
        }

        # Store the (reference to) outputs of completed tasks
        task_outputs: "dict[str, list[Any] | ray.ObjectRef | str]" = {}

        executed_tasks_count = 0
        total_tasks = len(execution_order)

        for task_name in execution_order:
            task_node = self._graph.get_node(task_name)
            task_node.input_data = self._resolve_input_data(
                task_node, initial_data, task_outputs
            )

            # Execute the task and store its output
            task_outputs[task_name] = self._execute_task(
                task_node, task_node.input_data, global_kwargs
            )
            executed_tasks_count += 1
            logger.info(
                f"Workflow '{self.name}': {executed_tasks_count}/{total_tasks} tasks completed."
            )

        logger.info(f"Workflow '{self.name}': All tasks executed.")

        # Collect final materialized results for the return dictionary
        final_results = {}
        for task_name, node in all_nodes.items():
            if node.status == "completed":
                if has_ray and isinstance(node.results, ray.ObjectRef):
                    try:
                        final_results[task_name] = ray.get(node.results)
                        logger.debug(
                            f"Workflow '{self.name}': Materialized results for '{task_name}' from Ray object store."
                        )
                    except Exception as e:
                        logger.error(
                            f"Workflow '{self.name}': Failed to retrieve final results for '{task_name}': {e}"
                        )
                        final_results[task_name] = (
                            f"Error retrieving final results: {e}"
                        )
                else:
                    final_results[task_name] = node.results
            elif node.status == "failed":
                final_results[task_name] = {"status": "failed", "error": node.error}
            else:
                final_results[task_name] = {
                    "status": node.status,
                    "error": node.error,
                }  # Should not happen if all executed

        return final_results

    def get_task_status(self, task_name: str) -> dict[str, Any]:
        """
        Returns the current status of a specific task within the workflow.
        """
        node = self._graph.get_node(task_name)
        return {
            "status": node.status,
            "error": node.error,
        }
