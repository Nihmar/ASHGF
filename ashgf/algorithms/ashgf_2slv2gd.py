"""ASHGF-2SLV-2GD: dual gradient + exponential history decay.

Applies exponential decay to the gradient history buffer before each
update, so recent gradients weigh more than stale ones in the PCA
that drives the ASHGF direction mixing.
"""

from __future__ import annotations

import logging

from ashgf.algorithms.ashgf_2slv2g import ASHGF2SLV2G

logger = logging.getLogger(__name__)

__all__ = ["ASHGF2SLV2GD"]


class ASHGF2SLV2GD(ASHGF2SLV2G):
    """Dual-gradient with exponential history decay.

    Parameters
    ----------
    history_decay : float
        Multiplier applied to the gradient buffer before each new entry.
        Default ``0.95``.
    **kwargs :
        Passed to :class:`ASHGF2SLV2G`.
    """

    kind = "ASHGF2SLV2GD"

    def __init__(self, history_decay: float = 0.95, **kwargs) -> None:
        super().__init__(**kwargs)
        self._history_decay = history_decay

    def grad_estimator(self, x, f):
        self._G_buffer *= self._history_decay
        return super().grad_estimator(x, f)
