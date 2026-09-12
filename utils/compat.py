"""Small compatibility helpers for the supported Python runtimes."""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Tuple


def strict_zip(*iterables: Iterable[Any]) -> Iterator[Tuple[Any, ...]]:
    """Yield tuples like ``zip(..., strict=True)`` on Python 3.9.

    The iterator raises ``ValueError`` as soon as it observes unequal input
    lengths.  Iterators are advanced from left to right so an earlier
    exhaustion does not consume a value from later inputs.
    """

    iterators = tuple(iter(value) for value in iterables)
    if not iterators:
        return
    while True:
        values = []
        try:
            values.append(next(iterators[0]))
        except StopIteration:
            for iterator in iterators[1:]:
                try:
                    next(iterator)
                except StopIteration:
                    continue
                raise ValueError("strict_zip() arguments have different lengths")
            return
        for iterator in iterators[1:]:
            try:
                values.append(next(iterator))
            except StopIteration:
                raise ValueError("strict_zip() arguments have different lengths")
        yield tuple(values)


__all__ = ["strict_zip"]
