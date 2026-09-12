"""CONNECT-4 target-blind inference entry point.

The runtime accepts structural conditioning only. Paired held-out targets and
metrics remain isolated to the post-seal evaluator.
"""
from __future__ import annotations

from .runtime import POSTSEAL_EVALUATOR_GUIDANCE, main


if __name__ == "__main__":
    main()
