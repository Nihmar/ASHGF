"""Logging configuration utilities."""

from __future__ import annotations

import logging
import sys


def configure_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    stream: str = "stderr",
) -> None:
    """Configure the root logger for the ashgf package.

    Parameters
    ----------
    level : int
        Logging level (e.g., ``logging.DEBUG``, ``logging.INFO``).
    format_string : str or None
        Custom format string. If None, a default is used.
    stream : str
        Either ``"stderr"`` or ``"stdout"``.
    """
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)-7s] %(name)s | %(message)s"

    handler = logging.StreamHandler(sys.stderr if stream == "stderr" else sys.stdout)
    handler.setFormatter(logging.Formatter(format_string, datefmt="%H:%M:%S"))

    ashgf_logger = logging.getLogger("ashgf")
    ashgf_logger.setLevel(level)
    ashgf_logger.handlers.clear()
    ashgf_logger.addHandler(handler)
    ashgf_logger.propagate = False
