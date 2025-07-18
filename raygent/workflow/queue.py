from typing import Any, Generic, TypeVar, override

from ray.util.queue import Queue

T = TypeVar("T")


class BoundedQueue(Queue, Generic[T]):
    """Thin wrapper around `ray.util.queue.Queue` enforcing `maxsize`.

    Ray's `Queue` already supports a `maxsize` parameter for built-in
    back-pressure (producers block once the queue is full).  This subclass
    merely makes `maxsize` a **mandatory** positional argument so edges cannot
    accidentally be created unbounded.
    """

    def __init__(
        self, maxsize: int, *, actor_options: dict[str, Any] | None = None
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        super().__init__(maxsize=maxsize, actor_options=actor_options or {})

    @override
    def __repr__(self) -> str:
        return f"<BoundedQueue maxsize={self.maxsize} size={self.qsize()}>"
