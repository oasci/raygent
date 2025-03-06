from typing import Any, Generator

from collections.abc import Callable

import ray

from raygent.savers import Saver
from raygent.worker import ray_worker


class TaskManager:
    """
    A manager class for handling task submissions and result collection using Ray's Task Parallelism.
    """

    def __init__(
        self,
        task_class: Callable[[], Any],
        n_cores: int = -1,
        use_ray: bool = False,
        n_cores_worker: int = 1,
    ) -> None:
        """Initializes the TaskManager.

        Creates a new TaskManager instance configured for either sequential or
        parallel task execution using the specified task class.

        Args:
            task_class: A callable that returns an instance with a `run` method for
                processing each item.
            n_cores: Number of parallel tasks to run. If <= 0, uses all available CPUs.
                Default is -1 (use all available cores).
            use_ray: Flag to determine if Ray should be used for parallel execution.
                If False, runs tasks sequentially. Default is False.
            n_cores_worker: The number of cores allocated for each worker.
                Default is 1.
        """
        self.task_class = task_class
        """
        A callable that returns a Task instance.

        This callable must return an object that implements the Task interface,
        with `run`, `process_item`, and/or `process_items` methods. It is invoked
        to create a new Task instance for each worker when using Ray, or once for
        sequential processing.

        Example:
            ```python
            def create_analyzer_task():
                return TextAnalyzerTask()

            manager = TaskManager(create_analyzer_task)
            ```
        """

        self.use_ray = use_ray
        """
        Boolean flag controlling whether to use Ray for parallel execution.

        When True, tasks are distributed across multiple cores or machines using Ray.
        When False, tasks are executed sequentially in the current process.

        Set this to True for computationally intensive workloads that can benefit
        from parallelization, and False for debugging or when the overhead of
        distributing tasks outweighs the benefits.

        Example:
            ```python
            # Sequential processing (for debugging)
            manager = TaskManager(MyTask, use_ray=False)

            # Parallel processing (for production)
            manager = TaskManager(MyTask, use_ray=True, n_cores=8)
            ```
        """

        self.n_cores = n_cores
        """
        The total number of CPU cores available for parallel execution.

        This value determines the overall parallelism level when `use_ray=True`.
        A value of -1 or any negative number will use all available CPU cores
        on the system. For specific resource allocation, set to a positive integer.

        For cluster environments, this represents the total cores available
        across all nodes.

        Example:
            ```python
            # Use all available cores
            manager = TaskManager(MyTask, use_ray=True, n_cores=-1)

            # Use exactly 4 cores
            manager = TaskManager(MyTask, use_ray=True, n_cores=4)
            ```
        """

        self.n_cores_worker = n_cores_worker
        """
        The number of CPU cores allocated to each worker process.

        This controls how many cores each task instance can utilize. Increase this
        value for compute-intensive tasks that can leverage multiple cores per task,
        or keep at 1 for maximum parallelism across tasks.

        The effective parallelism is determined by n_cores // n_cores_worker.

        Example:
            ```python
            # Each task gets 1 core (maximum task parallelism)
            manager = TaskManager(SimpleTask, use_ray=True, n_cores_worker=1)

            # Each task gets 2 cores (good for moderately parallel tasks)
            manager = TaskManager(ComputeTask, use_ray=True, n_cores_worker=2)

            # Each task gets 4 cores (for tasks with internal parallelism)
            manager = TaskManager(ParallelTask, use_ray=True, n_cores_worker=4)
            ```
        """

        self.futures: list[ray.ObjectRef] = []  # type: ignore
        """
        A list of Ray object references for currently submitted tasks.

        This attribute tracks all actively running but not yet completed tasks when
        using Ray. It is used internally to manage the pool of workers and to ensure
        that the number of concurrent tasks does not exceed max_concurrent_tasks.

        This list is empty when not using Ray or when no tasks are currently running.
        Users typically do not need to interact with this attribute directly.
        """

        self.results: list[Any] = []
        """
        A list storing the results of all completed tasks.

        This attribute accumulates the outputs from all Task.run calls, maintaining
        the order in which tasks complete (which may differ from submission order
        when using Ray). Results are added here after tasks complete and optionally
        after being processed by a saver.

        Access this list using the get_results() method after task submission.

        Example:
            ```python
            manager = TaskManager(MyTask)
            manager.submit_tasks(items)
            results = manager.get_results()  # Access the contents of this attribute
            ```
        """

        self.saver: Saver | None = None
        """
        An optional Saver instance for persisting results.

        When provided, this Saver is used to save results at intervals specified by
        save_interval. This allows for checkpointing and persistence of intermediate
        results during long-running computations.

        The saver must implement the Saver interface with a save(data) method.
        Set during submit_tasks() and not during initialization.

        Example:
            ```python
            manager = TaskManager(MyTask)
            saver = HDF5Saver("results.h5")
            manager.submit_tasks(items, saver=saver, save_interval=100)
            ```
        """

        self.save_interval: int = 1
        """
        The number of results to accumulate before invoking the saver.

        This controls how frequently results are saved when a saver is provided.
        Lower values reduce memory usage but may increase I/O overhead, while higher
        values can improve I/O efficiency at the cost of increased memory usage.

        Set during submit_tasks() and not during initialization. Default is 1.

        Example:
            ```python
            # Save every 100 results (good balance)
            manager.submit_tasks(items, saver=my_saver, save_interval=100)

            # Save every 1000 results (reduce I/O, more memory usage)
            manager.submit_tasks(items, saver=my_saver, save_interval=1000)
            ```
        """

        self.save_kwargs: dict[str, Any] = dict()
        """
        Additional keyword arguments passed to the saver's save method.

        This dictionary contains any extra parameters needed by the saver when
        saving results. It can include file paths, database connections, or
        other configuration options specific to the saver implementation.

        Example:
            ```python
            # Set up a manager with save options
            manager = TaskManager(MyTask)
            manager.save_kwargs = {
                "compression": "gzip",
                "append": True,
                "dtype": "float32"
            }
            manager.submit_tasks(items, saver=my_saver)
            ```
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
            * CPU-bound tasks benefit from higher n_cores_worker values
            * I/O-bound tasks typically work better with n_cores_worker=1 and
                higher max_concurrent_tasks

        Examples:
            ```python
            # With 8 total cores and 2 cores per worker
            manager = TaskManager(MyTask, n_cores=8, n_cores_worker=2, use_ray=True)
            print(manager.max_concurrent_tasks)  # Output: 4

            # With 16 total cores and 4 cores per worker
            manager = TaskManager(MyTask, n_cores=16, n_cores_worker=4, use_ray=True)
            print(manager.max_concurrent_tasks)  # Output: 4

            # With 24 total cores and 1 core per worker (for I/O-bound tasks)
            manager = TaskManager(IOBoundTask, n_cores=24, n_cores_worker=1, use_ray=True)
            print(manager.max_concurrent_tasks)  # Output: 24
            ```
        """
        return max(1, int(self.n_cores // self.n_cores_worker))

    def task_generator(
        self, items: list[Any], chunk_size: int
    ) -> Generator[Any, None, None]:
        """
        Splits a list of items into smaller chunks and yields each chunk for processing.

        This generator takes a list of items and partitions it into sublists (chunks) where each
        chunk contains up to `chunk_size` items. This is useful for processing large datasets
        in smaller, manageable batches, whether processing sequentially or in parallel using Ray.

        Args:
            items: The complete list of items to be processed.
            chunk_size: The number of items to include in each chunk. If the total number
                of items is not evenly divisible by `chunk_size`, the final chunk will contain
                the remaining items.

        Yields:
            Each yielded value is a sublist of `items` containing up to
                `chunk_size` elements.

        Example:
            ```python
            >>> items = list(range(10))
            >>> list(self.task_generator(items, 3))
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
            ```
        """
        for start in range(0, len(items), chunk_size):
            end = min(start + chunk_size, len(items))
            yield items[start:end]

    def submit_tasks(
        self,
        items: list[Any],
        chunk_size: int = 100,
        saver: Saver | None = None,
        at_once: bool = False,
        save_interval: int = 100,
        kwargs_task: dict[str, Any] = dict(),
        kwargs_remote: dict[str, Any] = dict(),
    ) -> None:
        """Submits and processes tasks in parallel or serial mode with optional
        periodic saving.

        This method is the primary entrypoint for task execution in the `TaskManager`.
        It takes a list of items to process, chunks them into manageable batches,
        and either processes them sequentially or distributes them across workers
        using Ray, depending on the `TaskManager`'s configuration.

        The method also supports periodic saving of results through a provided
        [`Saver`][savers.core.Saver] instance, allowing for checkpointing and
        persistence of intermediate results during long-running computations.

        Args:
            items: A list of items to process. Each item will be passed to the task's
                process_item method individually or as part of a batch if
                `at_once=True`.
            chunk_size: Number of items per processing chunk. Larger values may improve
                performance but increase memory usage per worker.
            saver: An optional [`Saver`][savers.core.Saver] instance that implements
                the save method for persisting results. If provided, results will be
                saved according to `save_interval`.
            at_once: If `True`, each chunk is processed as a batch by the task's
                [`process_items`][task.Task.process_items] method; otherwise,
                items are processed individually with
                [`process_item`][task.Task.process_item].
            save_interval: The number of results to accumulate before invoking the
                `saver`. Has no effect if `saver` is `None`.
            kwargs_task: Keyword arguments to pass to the task's run method.
                These can be used to customize task execution behavior.
            kwargs_remote: Keyword arguments to pass to Ray's remote function options.
                Only used when `use_ray=True`. Can include options like `max_retries`,
                `num_gpus`, etc., but not `num_cpus` (use
                [`n_cores_worker`][manager.TaskManager.n_cores_worker] and
                [`n_cores`][manager.TaskManager.n_cores] instead).

        Returns:
            None. Results are stored internally and can be retrieved using
                [`get_results()`][manager.TaskManager.get_results].

        Raises:
            ValueError: If the `saver`'s save method raises an exception.
            ImportError: If Ray is requested but not installed.

        Notes:
            -   When `use_ray=True` in the TaskManager, this method leverages Ray for
            parallel execution across cores or machines.
            -   When `use_ray=False`, tasks are processed sequentially in the
                current process.
            -   When a `saver` is provided, results are saved periodically according to
                `save_interval`, reducing memory usage for long-running tasks.

        Examples:
            Sequential processing:

            ```python
            # Create a task manager for sequential processing
            manager = TaskManager(NumberSquareTask, use_ray=False)

            # Process items without saving results
            manager.submit_tasks([1, 2, 3, 4, 5])
            results = manager.get_results()  # [1, 4, 9, 16, 25]
            ```

            Parallel processing with Ray:

            ```python
            # Create a task manager with Ray for parallel processing
            manager = TaskManager(TextProcessorTask, use_ray=True, n_cores=8)

            # Process a large list of items in chunks of 50
            manager.submit_tasks(
                text_documents,
                chunk_size=50,
                kwargs_task={"min_length": 10, "language": "en"}
            )
            results = manager.get_results()
            ```

            Processing with periodic saving:

            ```python
            # Create a task manager and a saver for results
            manager = TaskManager(DataAnalysisTask, use_ray=True)
            saver = HDF5Saver("results.h5", dataset_name="analysis_results")

            # Process items with saving every 1000 results
            manager.submit_tasks(
                large_dataset,
                chunk_size=200,
                saver=saver,
                save_interval=1000,
                kwargs_remote={"max_retries": 3}
            )
            ```
        """
        self.saver = saver
        self.save_interval = save_interval

        task_gen = self.task_generator(items, chunk_size)

        if self.use_ray:
            self._submit_ray(task_gen, at_once, kwargs_task, kwargs_remote)
        else:
            self._submit(task_gen, at_once, kwargs_task)

    def _submit(
        self,
        task_gen: Generator[Any, None, None],
        at_once: bool = False,
        kwargs_task: dict[str, Any] = dict(),
    ) -> None:
        """
        Handles task submission and ordered result collection in sequential mode.

        This method processes task chunks one at a time in the current process
        (i.e., without parallelization via Ray). For each chunk produced by the
        [`task_gen`][manager.TaskManager.task_generator] generator, a new
        task instance is created from [`self.task_class`][task.Task] and its
        [`run`][task.Task.run] method is invoked. Depending on
        the `at_once` flag, the chunk is processed either as a batch
        (using [`process_items`][task.Task.process_items]) or
        item by item (using [`process_item`][task.Task.process_item]).

        As each chunk is processed, the resulting items are appended to a temporary
        list. If a `Saver`(`self.saver`) is provided and the number of accumulated
        results meets or exceeds the specified `save_interval`, the Saver's `save`
        method is called to persist a portion of the results, and the
        saved results are moved into the final results list (`self.results`).

        After processing all chunks, any remaining results are saved (if applicable)
        and then appended to the final results list.

        Args:
            task_gen: Generator yielding chunks of items to process.
            at_once: If True, processes the entire chunk at once using the task's
                batch processing method; otherwise, processes each item individually.
            kwargs_task: Additional keyword arguments to pass to the task's
                `run` method.

        Returns:
            None. The processed results are stored in the instance attribute
                `self.results`.
        """
        results = []
        for chunk in task_gen:
            results_chunk = self.task_class().run(chunk, at_once=at_once, **kwargs_task)
            results.extend(results_chunk)

            if self.saver and len(results) >= self.save_interval:
                self.saver.save(results[: self.save_interval], **self.save_kwargs)
                self.results.extend(results[: self.save_interval])
                results = results[self.save_interval :]

        if self.saver and len(results) > 0:
            self.saver.save(results, **self.save_kwargs)
        self.results.extend(results)

    def _submit_ray(
        self,
        task_gen: Generator[Any, None, None],
        at_once: bool = False,
        kwargs_task: dict[str, Any] = dict(),
        kwargs_remote: dict[str, Any] = dict(),
    ) -> None:
        """
        Handles task submission and ordered result collection using Ray for parallel
        execution.

        This method iterates over a generator of task chunks and submits each chunk as
        a separate Ray task.

        To ensure that the final results are in the same order as the input items,
        each submitted task is tagged with a unique chunk index. A mapping is
        maintained between the string representation of each Ray ObjectRef and its
        corresponding chunk index. As tasks complete,
        their results—which may be a list of items—are stored in a temporary
        dictionary using their chunk index.

        Once contiguous chunks are available (starting from the first chunk),
        these chunks are flushed into a final results list in order.
        Additionally, if a Saver is provided, results are periodically saved
        when the number of accumulated results reaches a specified interval.

        Args:
            task_gen: A generator that yields chunks of items to process.
            at_once: If True, each task processes the entire chunk at once using the batch
                processing method; otherwise, each item in the chunk is processed individually.
                Defaults to False.
            kwargs_task: Additional keyword arguments to pass to the task's
                processing function. Defaults to an empty dictionary.
            kwargs_remote: Additional keyword arguments to pass to Ray's remote
                function options. Do not include 'num_cpus' here; set that using `n_cores_worker` instead.
                Defaults to an empty dictionary.

        Returns:
            None. The final, ordered results are stored in the instance attribute
                `results`.
        """
        if not ray.is_initialized():
            ray.init()

        if self.n_cores < 0:
            self.n_cores = int(ray.available_resources().get("CPU", 1))

        # Dictionary to hold results per chunk (key: chunk index, value: list of results)
        results_chunks: dict[int, list[Any]] = {}
        # Pointer to the next chunk index that should be flushed into the final results list
        next_chunk_index = 0
        # For mapping ObjectRef to its chunk index
        indices_future: dict[str, int] = {}
        chunk_index = 0
        final_results: list[Any] = []  # Temporary buffer for unsaved results

        # Submit tasks
        for chunk in task_gen:
            # If we're at max concurrency, wait for one to finish
            if len(self.futures) >= self.max_concurrent_tasks:
                done_futures, self.futures = ray.wait(self.futures, num_returns=1)
                # Get the chunk index for this finished task
                finished_index = indices_future.pop(str(done_futures[0]))
                results_chunk = ray.get(done_futures[0])
                # Save the result chunk in the dictionary
                results_chunks[finished_index] = results_chunk

                # Flush any contiguous finished chunks to final_results
                while next_chunk_index in results_chunks:
                    final_results.extend(results_chunks.pop(next_chunk_index))
                    next_chunk_index += 1

                    # Optionally, save periodically if a saver is provided
                    if self.saver and len(final_results) >= self.save_interval:
                        self.saver.save(
                            final_results[: self.save_interval], **self.save_kwargs
                        )
                        self.results.extend(final_results[: self.save_interval])
                        final_results = final_results[self.save_interval :]

            # Submit a new task to Ray
            future = ray_worker.options(
                num_cpus=self.n_cores_worker, **kwargs_remote
            ).remote(self.task_class, chunk, at_once, **kwargs_task)
            self.futures.append(future)
            indices_future[str(future)] = chunk_index
            chunk_index += 1

        # Collect remaining futures
        while self.futures:
            done_futures, self.futures = ray.wait(self.futures, num_returns=1)
            finished_index = indices_future.pop(str(done_futures[0]))
            results_chunk = ray.get(done_futures[0])
            results_chunks[finished_index] = results_chunk

            # Flush contiguous finished chunks
            while next_chunk_index in results_chunks:
                final_results.extend(results_chunks.pop(next_chunk_index))
                next_chunk_index += 1
                if self.saver and len(final_results) >= self.save_interval:
                    self.saver.save(
                        final_results[: self.save_interval], **self.save_kwargs
                    )
                    self.results.extend(final_results[: self.save_interval])
                    final_results = final_results[self.save_interval :]

        # Save any remaining results
        if self.saver and final_results:
            self.saver.save(final_results, **self.save_kwargs)
        self.results.extend(final_results)

    def get_results(self) -> list[Any]:
        """Retrieves all collected results from completed tasks.

        This method provides access to the accumulated results that have been
        collected from all tasks that have been submitted and completed through
        the `TaskManager`. Results are stored in the order they were processed.

        If a saver was provided during task submission, the results returned by
        this method will be the same as those that were passed to the saver.

        Returns:
            A list containing the results from all completed tasks. The structure
                of individual results depends on what was returned by the Task's
                process_item or process_items methods.

        Notes:
            - This method simply returns the internal results list attribute
            and does not perform any additional processing or computation.
            - Results are available immediately after tasks complete, whether
            run sequentially or in parallel via Ray.
            - If no tasks have been submitted or completed, an empty list is returned.

        Examples:
            ```python
            # Create a task manager and submit tasks
            manager = TaskManager(NumberSquareTask)
            manager.submit_tasks([1, 2, 3, 4, 5])

            # Retrieve and use results
            results = manager.get_results()
            # results = [1, 4, 9, 16, 25]

            # Results can be processed further
            sum_of_squares = sum(results)  # 55
            ```
        """
        return self.results
