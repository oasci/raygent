from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar, Unpack

from collections.abc import Generator, Iterable, Mapping

try:
    import ray

    from raygent.worker import ray_worker

    has_ray = True
except ImportError:
    has_ray = False
from loguru import logger

from raygent.batch import Ts, batch_generator
from raygent.results.handlers import ResultsHandler

if TYPE_CHECKING:
    from raygent import Task

P = ParamSpec("P")
T = TypeVar("T")


class TaskRunner(Generic[P, T]):
    """
    A runner class for handling task submissions and result handling using serial
    or parallel computation.
    """

    def __init__(
        self,
        task_cls: "type[Task[P, T]]",
        handler_cls: type[ResultsHandler[T]],
        n_cores: int = -1,
        in_parallel: bool = False,
        n_cores_worker: int = 1,
    ) -> None:
        """
        Args:
            task_cls: A class that is type [`Task`][task.Task].
            handler_cls: Class that collects, processes, and handles all
                [`Result`][results.result.Result]s after calling
                [`run_batch`][task.Task.run_batch] on `task`.
            n_cores: Number of parallel tasks to run. If <= 0, uses all available CPUs.
            in_parallel: Flag to determine if Ray should be used for parallel execution.
            n_cores_worker: The number of cores allocated for each worker if
                `in_parallel` is `True`.
        """

        self.task_cls: "type[Task[P, T]]" = task_cls
        """
        A class that follows the [`Task`][task.Task] protocol.
        """

        self.handler_cls: type[ResultsHandler[T]] = handler_cls
        """
        Class that collects, processes, and handles all
        [`Result`][results.result.Result]s after calling
        [`run_batch`][task.Task.run_batch] on `task`. Defaults to
        [`ResultsCollector`][results.handlers.collector.ResultsCollector].
        """

        self.handler: ResultsHandler[T] = self.handler_cls()

        assert isinstance(in_parallel, bool), "in_parallel must be a bool"
        if in_parallel is True and not has_ray:
            raise ImportError("Requested to use ray, but ray is not installed.")

        self.in_parallel: bool = in_parallel
        """
        Boolean flag controlling whether to use Ray for parallel execution.

        When `True`, tasks are distributed across multiple cores or machines using Ray.
        When `False`, tasks are executed sequentially in the current process.

        Example:
            ```python
            # Serial processing
            runner = TaskRunner(MyTask, ResultsCollector, in_parallel=False)

            # Parallel processing
            runner = TaskRunner(MyTask, ResultsCollector, in_parallel=True)
            ```
        """

        if isinstance(n_cores, float):
            n_cores: int = int(n_cores)
        assert isinstance(n_cores, int), "n_cores must be an int"

        self.n_cores: int = n_cores
        """
        The total number of CPU cores available for parallel execution.

        This value determines the overall parallelism level when `in_parallel=True`.
        A value of `-1` or any negative number will use all available CPU cores
        on the system. For specific resource allocation, set to a positive integer.

        For cluster environments, this represents the total cores available
        across all nodes.

        Example:
            ```python
            # Use all available cores
            runner = TaskRunner(MyTask, ResultsCollector, in_parallel=True, n_cores=-1)

            # Use up to 4 cores
            runner = TaskRunner(MyTask, ResultsCollector, in_parallel=True, n_cores=4)
            ```
        """

        if isinstance(n_cores_worker, float):
            n_cores_worker: int = int(n_cores_worker)
        assert isinstance(n_cores_worker, int), "n_cores_worker must be an int"

        self.n_cores_worker: int = n_cores_worker
        """
        The number of CPU cores allocated to each worker process.

        This controls how many cores each task instance can utilize. Increase this
        value for compute-intensive tasks that can leverage multiple cores per task,
        or keep at `1` for maximum parallelism across tasks.

        The effective parallelism is determined by `n_cores // n_cores_worker`.

        Example:
            ```python
            # Each task gets 1 core (maximum task parallelism)
            runner = TaskRunner(SimpleTask, ResultsCollector, in_parallel=True, n_cores_worker=1)

            # Each task gets 2 cores (good for moderately parallel tasks)
            runner = TaskRunner(ComputeTask, ResultsCollector, in_parallel=True, n_cores_worker=2)

            # Each task gets 4 cores (for tasks with internal parallelism)
            runner = TaskRunner(ParallelTask, ResultsCollector, in_parallel=True, n_cores_worker=4)
            ```
        """

        self.futures: "list[ray.ObjectRef]" = []
        """
        A list of Ray object references for currently submitted tasks.

        This attribute tracks all actively running but not yet completed tasks when
        using Ray. It is used internally to manage the pool of workers and to ensure
        that the number of concurrent tasks does not exceed max_concurrent_tasks.

        This list is empty when not using Ray or when no tasks are currently running.
        Users typically do not need to interact with this attribute directly.
        """

    @property
    def max_concurrent_tasks(self) -> int:
        """Maximum number of concurrent tasks that can run in parallel.

        This property calculates the maximum number of tasks that can be executed
        concurrently based on the total available CPU cores (n_cores) and the
        number of cores allocated per worker (n_cores_worker).

        The value represents the task parallelism limit when using Ray for
        distributed execution. It ensures efficient resource utilization by
        preventing over-subscription of CPU resources.

        Returns:
            An integer representing the maximum number of concurrent tasks.
                Always returns at least 1, even if calculations result in a lower value.

        Notes:
            - This property is primarily used internally by the `_submit_ray` method
            to manage the worker pool size.
            - The calculation is done by dividing total cores by cores per worker
            and flooring the result.
            - Setting n_cores_worker appropriately is important for tasks with
            different computational profiles:
            - CPU-bound tasks benefit from higher n_cores_worker values
            - I/O-bound tasks typically work better with n_cores_worker=1 and
                higher max_concurrent_tasks

        Examples:
            ```python
            # With 8 total cores and 2 cores per worker
            runner = TaskRunner(
                MyTask, ResultsCollector, n_cores=8, n_cores_worker=2, in_parallel=True
            )
            print(runner.max_concurrent_tasks)  # Output: 4

            # With 16 total cores and 4 cores per worker
            runner = TaskRunner(
                MyTask, ResultsCollector, n_cores=16, n_cores_worker=4, in_parallel=True
            )
            print(runner.max_concurrent_tasks)  # Output: 4

            # With 24 total cores and 1 core per worker (for I/O-bound tasks)
            runner = TaskRunner(
                IOBoundTask,
                ResultsCollector,
                n_cores=24,
                n_cores_worker=1,
                in_parallel=True,
            )
            print(runner.max_concurrent_tasks)  # Output: 24
            ```
        """
        n_tasks: int = max(1, int(self.n_cores // self.n_cores_worker))
        logger.debug(f"Total cores: {self.n_cores}")
        logger.debug(f"Cores per worker: {self.n_cores_worker}")
        logger.debug(f"Maximum number of concurrent tasks: {n_tasks}")
        return n_tasks

    def submit_tasks(
        self,
        *data_streams: Unpack[tuple[Iterable[Any]]],
        batch_size: int = 10,
        prebatched: bool = False,
        args_task: tuple[object] | None = None,
        kwargs_task: Mapping[str, object] | None = None,
        args_remote: tuple[object] | None = None,
        kwargs_remote: Mapping[str, object] | None = None,
    ) -> ResultsHandler[T]:
        """Submits and processes tasks in serial or parallel mode.

        This method is the primary entrypoint for task execution in the `TaskRunner`.
        It takes data and batches, processes, and handles all results. Note that
        `self.handler` is set here and will overwrite if present.

        Args:
            data: Data to process.
            batch_size: Number of items per processing batch. Larger values may improve
                performance but increase memory usage per worker.
            prebatched: `data` is already an iterable of batches.
            args_task: Arguments for [`run_batch`][task.Task.run_batch].
            kwargs_task: Keyword arguments for [`run_batch`][task.Task.run_batch].
            args_remote: Arguments for Ray's remote function options.
            kwargs_remote: Keyword arguments to pass to Ray's remote function options.
                Only used when `in_parallel=True`. Can include options like `max_retries`,
                `num_gpus`, etc., but not `num_cpus` (use
                [`n_cores_worker`][runner.TaskRunner.n_cores_worker] and
                [`n_cores`][runner.TaskRunner.n_cores] instead).

        Returns:
            The [results handler][runner.TaskRunner.handler_cls] after processing
                all batches.

        Notes:
            -   When `in_parallel=True` in the TaskRunner, this method leverages Ray for
            parallel execution across cores or machines.
            -   When `in_parallel=False`, tasks are processed sequentially in the
                current process.

        Examples:
            Sequential processing:

            ```python
            # Create a task runner for sequential processing
            runner = TaskRunner(NumberSquareTask, ResultsCollector, in_parallel=False)

            # Process items without saving results
            handler = runner.submit_tasks([1, 2, 3, 4, 5])
            results = handler.get()  # [1, 4, 9, 16, 25]
            ```

            Parallel processing with Ray:

            ```python
            # Create a task runner with Ray for parallel processing
            runner = TaskRunner(
                TextProcessorTask, ResultsCollector, in_parallel=True, n_cores=8
            )

            # Process a large list of items in batches of 50
            handler = runner.submit_tasks(
                text_documents,
                batch_size=50,
                kwargs_task={"min_length": 10, "language": "en"},
            )
            results = handler.get()
            ```
        """
        self.handler = self.handler_cls()

        logger.info(f"Submitting tasks with batch_size of {batch_size}")
        batch_gen = batch_generator(
            *data_streams, batch_size=batch_size, prebatched=prebatched
        )

        if self.in_parallel:
            self._submit_ray(
                batch_gen, args_task, kwargs_task, args_remote, kwargs_remote
            )
        else:
            self._submit(batch_gen, args_task, kwargs_task)

        self.handler.finalize()
        return self.handler

    def _submit(
        self,
        batch_gen: Generator[tuple[int, tuple[*Ts]], None, None],
        args_task: tuple[object] | None = None,
        kwargs_task: Mapping[str, object] | None = None,
    ) -> None:
        """
        Handles task submission and ordered result collection in sequential mode.

        This method processes task batches one at a time in the current process
        (i.e., without parallelization via Ray). For each batch produced by the
        [`batch_gen`][runner.TaskRunner.batch_gen] generator, a new
        task instance is created from [`self.task_cls`][task.Task] and its
        [`run_batch`][task.Task.run_batch] method is invoked.

        Args:
            batch_gen: Generator yielding batches of items to process.
            kwargs_task: Additional keyword arguments to pass to the task's
                `run_batch` method.

        Returns:
            None. The processed results are stored in the instance attribute
                `self.results`.
        """
        if args_task is None:
            args_task = tuple()
        if kwargs_task is None:
            kwargs_task = {}
        logger.debug("Running tasks in serial")
        for index, batch_item_args in batch_gen:
            task_instance = self.task_cls()
            results_batch = task_instance.run_batch(
                index, *batch_item_args, *args_task, **kwargs_task
            )
            self.handler.add_result(results_batch)
            self.handler.save()

    def _submit_ray(
        self,
        batch_gen: Generator[tuple[int, tuple[*Ts]], None, None],
        args_task: tuple[object] | None = None,
        kwargs_task: Mapping[str, object] | None = None,
        args_remote: tuple[object] | None = None,
        kwargs_remote: Mapping[str, object] | None = None,
    ) -> None:
        """
        Handles task submission and ordered result collection using Ray for parallel
        execution.

        This method iterates over a generator of task batches and submits each batch as
        a separate Ray task.

        To ensure that the final results are in the same order as the input items,
        each submitted task is tagged with a unique batch index. A mapping is
        maintained between the string representation of each Ray ObjectRef and its
        corresponding batch index. As tasks complete,
        their results—which may be a list of items—are stored in a temporary
        dictionary using their batch index.

        Once contiguous batches are available (starting from the first batch),
        these batches are flushed into a final results list in order.
        Additionally, if a Saver is provided, results are periodically saved
        when the number of accumulated results reaches a specified interval.

        Args:
            batch_gen: A generator that yields batches of items to process.
            kwargs_task: Additional keyword arguments to pass to the task's
                processing function. Defaults to an empty dictionary.
            kwargs_remote: Additional keyword arguments to pass to Ray's remote
                function options. Do not include 'num_cpus' here; set that using `n_cores_worker` instead.
                Defaults to an empty dictionary.

        Returns:
            None. The final, ordered results are stored in the instance attribute
                `results`.
        """
        # pyright: reportUnknownMemberType=false, reportPossiblyUnboundVariable=false
        # pyright: reportUnknownArgumentType=false
        if args_task is None:
            args_task = tuple()
        if kwargs_task is None:
            kwargs_task = {}
        if args_remote is None:
            args_remote = tuple()
        if kwargs_remote is None:
            kwargs_remote = {}
        logger.debug("Running tasks in parallel with Ray")
        if not ray.is_initialized():
            logger.info("Initializing Ray.")
            ray.init()

        if self.n_cores < 0:
            n_cores = int(ray.available_resources().get("CPU", 1))
            logger.info("Ray will be using {} cores", n_cores)
            self.n_cores = n_cores

        # Submit tasks and process completed ones as we go.
        for index, batch_item_args in batch_gen:
            # If the maximum concurrency is reached, wait for one task to finish.
            if len(self.futures) >= self.max_concurrent_tasks:
                done_futures, self.futures = ray.wait(self.futures, num_returns=1)
                finished_future = done_futures[0]
                result = ray.get(finished_future)
                self.handler.add_result(result)

            # Submit a new task to Ray.
            logger.debug(f"Submitting Ray task which index of {index}")
            future = ray_worker.options(
                num_cpus=self.n_cores_worker, *args_remote, **kwargs_remote
            ).remote(self.task_cls, index, *batch_item_args, *args_task, **kwargs_task)
            self.futures.append(future)

        # Process any remaining futures.
        while self.futures:
            done_futures, self.futures = ray.wait(self.futures, num_returns=1)
            finished_future = done_futures[0]
            result = ray.get(finished_future)
            self.handler.add_result(result)

    def get_handler(self) -> ResultsHandler[T]:
        """Returns the `ResultsHandler`.

        Returns:
            The `ResultsHandler`.
        """
        return self.handler
