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

        Args:
            task_class: A callable that returns an instance with a `run` method for
                processing each item.
            n_cores: Number of parallel tasks to run. If <= 0, uses all available CPUs.
            use_ray: Flag to determine if Ray should be used. If False, runs tasks sequentially.
            n_cores_worker: The number of cores allocated for each worker.
        """
        self.task_class = task_class
        """
        A callable that returns an instance with a `run` method for
        processing each item.
        """

        self.use_ray = use_ray
        """
        Flag to determine if Ray should be used. If `False`, runs tasks sequentially.
        """

        self.n_cores = n_cores
        """
        The total number of cores available. If set to `-1` or any value less than or
        equal to `0`, all available CPU cores are utilized.
        """

        self.n_cores_worker = n_cores_worker
        """
        The number of cores allocated for each worker.
        """

        self.futures: list[ray.ObjectRef] = []  # type: ignore
        """
        A list of Ray object references representing the currently submitted but
        not yet completed tasks. This manages the pool of active workers.
        """

        self.results: list[Any] = []
        """
        A list that stores the results of all completed tasks. It aggregates the
        output returned by each worker.
        """

        self.saver: Saver | None = None
        """
        An optional Saver that takes a batch of results and performs a
        save operation. This can be used to persist intermediate results to
        disk, a database, or any other storage medium. If set to `None`, results are
        not saved automatically.
        """

        self.save_interval: int = 1
        """
        The number of results to accumulate before invoking `save_func`.
        When the number of collected results reaches this interval,
        `save_func` is called to handle the batch of results.
        """

        self.save_kwargs: dict[str, Any] = dict()
        """
        A dictionary of additional keyword arguments to pass to
        `save` when it is called. This allows for flexible configuration of the
        save operation, such as specifying file paths,
        database connections, or other parameters required by `save`.
        """

    @property
    def max_concurrent_tasks(self) -> int:
        """
        Maximum number of concurrent tasks if using ray.
        """
        return max(1, int(self.n_cores // self.n_cores_worker))

    def task_generator(
        self, items: list[Any], chunk_size: int
    ) -> Generator[Any, None, None]:
        """Generator that yields individual items from chunks.

        Args:
            items: A list of items to process.
            chunk_size: Number of items per chunk.

        Yields:
            Individual items to be processed.
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
        """Submits tasks using a generator and manages workers up to n_cores.

        Args:
            items: A list of items to process.
            chunk_size: Number of items per chunk.
            save_func: A callable that takes a list of results and saves them.
            at_once: If `True`, calls `process_items` to process all
                items at once; otherwise, processes them individually.
            save_interval: The number of results after which to invoke save_func.
            kwargs_task: Keyword arguments to pass into the task.
            kwargs_remote: Keyword arguments to pass into `ray.remote` options.
                `num_cpus` should not be included here, but set by changing
                [`TaskManager.n_cores_worker`][manager.TaskManager.n_cores_worker].
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
        """Handles task submission and result collection sequentially.

        Args:
            task_gen: Generator yielding tasks to process.
            at_once: If `True`, calls `process_items` to process all
                items at once; otherwise, processes them individually.
            kwargs_task: Keyword arguments to pass into the task.
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
        """Handles task submission and result collection using Ray.

        Args:
            task_gen: Generator yielding tasks to process.
            at_once: If `True`, calls `process_items` to process all
                items at once; otherwise, processes them individually.
            kwargs_task: Keyword arguments to pass into the task.
            kwargs_remote: Keyword arguments to pass into `ray.remote` options.
                `num_cpus` should not be included here, but set by changing
                [`TaskManager.n_cores_worker`][manager.TaskManager.n_cores_worker].
        """
        if not ray.is_initialized():
            ray.init()

        if self.n_cores < 0:
            self.n_cores = int(ray.available_resources().get("CPU", 1))

        results = []
        for chunk in task_gen:
            if len(self.futures) >= self.max_concurrent_tasks:
                # Wait for any worker to finish
                done_futures, self.futures = ray.wait(self.futures, num_returns=1)
                results_chunk = ray.get(done_futures[0])
                results.extend(results_chunk)

                if self.saver and len(results) >= self.save_interval:
                    self.saver.save(results[: self.save_interval], **self.save_kwargs)
                    self.results.extend(results[: self.save_interval])
                    results = results[self.save_interval :]

            # Submit new task to Ray
            future = ray_worker.options(
                num_cpus=self.n_cores_worker, **kwargs_remote
            ).remote(self.task_class, chunk, at_once, **kwargs_task)
            self.futures.append(future)

        # Collect remaining Ray futures
        while self.futures:
            done_futures, self.futures = ray.wait(self.futures, num_returns=1)
            results_chunk = ray.get(done_futures[0])
            results.extend(results_chunk)

            if self.saver and len(results) >= self.save_interval:
                self.saver.save(results[: self.save_interval], **self.save_kwargs)
                self.results.extend(results[: self.save_interval])
                results = results[self.save_interval :]

        if self.saver and results:
            self.saver.save(results, **self.save_kwargs)
        self.results.extend(results)

    def get_results(self) -> list[Any]:
        """
        Retrieves all collected results.

        Returns:
            A list of results from all completed tasks.
        """
        return self.results
