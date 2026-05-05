"""ASHGF-2SLV-α: zero random-direction mixing (alpha=0).

Forces the gradient-history direction mixing to use only the gradient
subspace (alpha = 0, k1 = k2 = 0).  No random directions are mixed in
after the warm-up phase, eliminating the noise from the SGES-style
sampling.  Also removes the warm-up (t=0) since the first gradient
immediately populates the history.
"""

from __future__ import annotations

import logging

from ashgf.algorithms.ashgf_2slv import ASHGF2SLV

logger = logging.getLogger(__name__)

__all__ = ["ASHGF2SLVA"]


class ASHGF2SLVA(ASHGF2SLV):
    """ASHGF-2SLV with alpha=0 (gradient-subspace only).

    Parameters
    ----------
    **kwargs :
        Passed to :class:`ASHGF2SLV`.
    """

    kind = "ASHGF2SLVA"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("k1", 0.0)
        kwargs.setdefault("k2", 0.0)
        kwargs.setdefault("alpha", 0.0)
        kwargs.setdefault("t", 1)
        super().__init__(**kwargs)
