"""Parity harness driving the frozen prototype in ``original_python_code/``.

The adapter imports the unmodified original modules through ``sys.path``, wraps
their objective evaluation for exact function-evaluation counting, and returns
structured :class:`OriginalRecord` objects comparable with the new package's
:class:`~ashgf.algorithms.RunResult`.
"""

from __future__ import annotations

from .original_runner import ORIGINAL_ALGORITHMS, OriginalRecord, run_original

__all__ = ["ORIGINAL_ALGORITHMS", "OriginalRecord", "run_original"]
