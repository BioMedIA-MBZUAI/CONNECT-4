"""Evaluation: synthetic-vs-real rs-fMRI metrics and visualisation.

Exports are loaded lazily so importing a provenance/security submodule (for
example :mod:`eval.postseal`) does not import synthesis-model dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULE = {
    "SynthesisMetricAccumulator": ".metrics",
    "compute_all": ".metrics",
    "plot_real_vs_synthetic": ".visualize",
    "plot_real_vs_synthetic_4d": ".visualize",
    "save_fmri_nifti": ".visualize",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "SynthesisMetricAccumulator",
    "compute_all",
    "plot_real_vs_synthetic",
    "plot_real_vs_synthetic_4d",
    "save_fmri_nifti",
]
