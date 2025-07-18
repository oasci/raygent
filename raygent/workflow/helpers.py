from typing import TypeVar, override

from raygent import Task

T = TypeVar("T")


class IdentityTask(Task):
    """Return the data unchanged."""

    @override
    def do(self, payload: T) -> T:
        return payload
