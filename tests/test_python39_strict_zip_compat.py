from __future__ import annotations

import ast
from pathlib import Path

import pytest

from utils.compat import strict_zip


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _CountingIterator:
    def __init__(self, values: list[int]) -> None:
        self._iterator = iter(values)
        self.calls = 0

    def __iter__(self) -> "_CountingIterator":
        return self

    def __next__(self) -> int:
        self.calls += 1
        return next(self._iterator)


def test_strict_zip_matches_equal_iterables_and_generators() -> None:
    assert list(strict_zip([1, 2], (3, 4))) == [(1, 3), (2, 4)]
    assert list(strict_zip((value for value in range(2)), "ab")) == [
        (0, "a"),
        (1, "b"),
    ]
    assert list(strict_zip()) == []


@pytest.mark.parametrize(
    "values",
    [([1], []), ([], [1]), ([1, 2], [3], [4, 5])],
)
def test_strict_zip_rejects_unequal_lengths(values: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="different lengths"):
        list(strict_zip(*values))


def test_strict_zip_does_not_overconsume_after_middle_exhaustion() -> None:
    first = _CountingIterator([1, 2])
    middle = _CountingIterator([3])
    last = _CountingIterator([4, 5, 6])
    iterator = strict_zip(first, middle, last)
    assert next(iterator) == (1, 3, 4)
    with pytest.raises(ValueError, match="different lengths"):
        next(iterator)
    assert [first.calls, middle.calls, last.calls] == [2, 2, 1]


def test_production_code_has_no_python310_zip_strict_keyword() -> None:
    offenders: list[str] = []
    for path in REPOSITORY_ROOT.rglob("*.py"):
        if "tests" in path.relative_to(REPOSITORY_ROOT).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "zip"
                and any(keyword.arg == "strict" for keyword in node.keywords)
            ):
                offenders.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")
    assert offenders == []
