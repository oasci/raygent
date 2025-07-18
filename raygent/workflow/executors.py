from typing import Any

from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager

try:
    import ray

    has_ray = True
except ImportError:
    has_ray = False


class InlineExecutor(Executor):
    """Minimal drop‑in Executor that runs tasks synchronously."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:  # type: ignore[override]  # noqa: D401,E501
        f: Future = Future()
        if f.set_running_or_notify_cancel():
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                f.set_exception(exc)
            else:
                f.set_result(result)
        return f

    def shutdown(self, wait: bool = True) -> None:
        pass  # nothing to clean up


if has_ray:

    class _RayFuture(Future):  # type: ignore[misc]
        """Adapter so a `ray.ObjectRef` behaves like a concurrent Future."""

        def __init__(self, obj_ref: "ray.ObjectRef"):
            super().__init__()
            self._obj_ref = obj_ref

        def result(self, timeout: float | None = None):
            return ray.get(self._obj_ref, timeout=timeout)

        def done(self) -> bool:  # noqa: D401
            return self._obj_ref.is_ready()

        def add_done_callback(self, fn):
            # Ray lacks callback hooks; poll in a thread if needed.
            raise NotImplementedError("Ray futures don't support callbacks")

    class _RayExecutor(Executor):
        """Thin adapter so Ray conforms to the Executor protocol."""

        def __init__(self, max_workers: int | None):
            self._max_workers = max_workers
            self._inflight: list[_RayFuture] = []

        def submit(self, fn, /, *args, **kwargs):
            obj_ref = _ray_wrapper.remote(fn, *args, **kwargs)
            fut = _RayFuture(obj_ref)
            self._inflight.append(fut)
            return fut

        def shutdown(self, wait: bool = True):
            if wait and self._inflight:
                ray.get([f._obj_ref for f in self._inflight])

    @ray.remote
    def _ray_wrapper(fn, *args, **kwargs):
        return fn(*args, **kwargs)


@contextmanager
def get_executor(parallel: bool, max_workers: int | None):
    """
    Context manager yielding an Executor.

    * If *parallel* is False → InlineExecutor (synchronous).
    * If *parallel* is True  → Ray (if installed) else ThreadPoolExecutor.
    """
    if parallel:
        if has_ray:
            if not ray.is_initialized():
                ray.init()
            exec_: Executor = _RayExecutor(max_workers)
        else:
            exec_ = ThreadPoolExecutor(max_workers=max_workers)
    else:
        exec_ = InlineExecutor()

    try:
        yield exec_
    finally:
        exec_.shutdown(wait=True)
