from typing import Any

from collections.abc import Generator, Iterable

try:
    import ray

    has_ray = True
except ImportError:
    has_ray = False


def _chain_iterables(*iterables: Iterable[Any]) -> Generator[Any, None, None]:
    """Chains multiple iterables (including Ray ObjectRefs) into a single generator."""
    for iterable in iterables:
        if has_ray and isinstance(iterable, ray.ObjectRef):
            yield from ray.get(iterable)
        else:
            yield from iterable
