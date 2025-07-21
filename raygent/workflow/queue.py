# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scienting Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.


from typing import Any, override

import uuid

from ray.util.queue import Queue


class BoundedQueue(Queue):
    """Thin wrapper around `ray.util.queue.Queue` enforcing `maxsize`.

    Ray's `Queue` already supports a `maxsize` parameter for built-in
    back-pressure (producers block once the queue is full). This subclass
    merely makes `maxsize` a mandatory positional argument so edges cannot
    accidentally be created unbounded.
    """

    def __init__(
        self, maxsize: int, *, actor_options: dict[str, Any] | None = None
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.uid: str = uuid.uuid4().hex[:8]
        super().__init__(maxsize=maxsize, actor_options=actor_options or {})

    @override
    def __repr__(self) -> str:
        return f"<BoundedQueue uid={self.uid}>"
