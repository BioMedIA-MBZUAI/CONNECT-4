"""Regression tests for the lightweight ``data`` package boundary."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_provenance_import_does_not_load_dataset_or_excel_stack() -> None:
    repository = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import data.provenance; "
                "assert 'data.dataset' not in sys.modules; "
                "assert 'openpyxl' not in sys.modules"
            ),
        ],
        cwd=repository,
        check=True,
    )


def test_public_exports_remain_available_lazily() -> None:
    import data

    assert data.Patchify3D.__name__ == "Patchify3D"
    assert data.Connect4Dataset.__name__ == "Connect4Dataset"
