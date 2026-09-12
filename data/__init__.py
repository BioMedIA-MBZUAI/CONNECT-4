"""Public data-loading exports.

Keep these imports lazy.  Python executes a package's ``__init__`` before it
loads any submodule, so eager dataset imports made lightweight modules such as
``data.provenance`` depend on every optional clinical-data dependency.
"""
from __future__ import annotations

from typing import Any


__all__ = ["Connect4Dataset", "Patchify3D"]


def __getattr__(name: str) -> Any:
    if name == "Connect4Dataset":
        from .dataset import Connect4Dataset

        return Connect4Dataset
    if name == "Patchify3D":
        from .patchify import Patchify3D

        return Patchify3D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
