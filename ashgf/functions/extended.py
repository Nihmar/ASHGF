"""Extended test functions for optimization benchmarks.

This module provides 19 standalone functions, each accepting a numpy array
``x`` and returning a scalar ``float``.  They are converted from the old
class-based ``Function`` implementation, with ``self.`` references and
``self.power(x)`` calls replaced by direct vectorised NumPy operations.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "extended_baele",
    "extended_bd1",
    "extended_cliff",
    "extended_denschnb",
    "extended_denschnf",
    "extended_feudenstein_and_roth",
    "extended_hiebert",
    "extended_himmelblau",
    "extended_maratos",
    "extended_penalty",
    "extended_psc1",
    "extended_quadratic_exponential_ep1",
    "extended_rosenbrock",
    "extended_tridiagonal_1",
    "extended_tridiagonal_2",
    "extended_trigonometric",
    "extended_white_and_holst",
    "generalized_rosenbrock",
    "generalized_white_and_holst",
]


# ---------------------------------------------------------------------------
# 1. extended_feudenstein_and_roth
# ---------------------------------------------------------------------------


def extended_feudenstein_and_roth(x: np.ndarray) -> float:
    """Extended Freudenstein & Roth function.

    Parameters
    ----------
    x : np.ndarray
        Input vector of even length.

    Returns
    -------
    float
        Sum of squared terms over pairs ``(x_p, x_d)``.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]

    inner = ((5.0 - x_d) * x_d - 2.0) * x_d
    term_1 = (-13.0 + x_p + inner) ** 2
    term_2 = (-29.0 + x_p + inner) ** 2

    return float(np.sum(term_1 + term_2))


# ---------------------------------------------------------------------------
# 2. extended_trigonometric
# ---------------------------------------------------------------------------


def extended_trigonometric(x: np.ndarray) -> float:
    """Extended Trigonometric function.

    Parameters
    ----------
    x : np.ndarray
        Input vector of arbitrary length ``n``.

    Returns
    -------
    float
        Sum of squared trigonometric terms.
    """
    n = len(x)
    cos_x = np.cos(x)

    term_1 = n - np.sum(cos_x)  # scalar
    term_2 = np.arange(1, n + 1) * (1.0 - cos_x)  # array
    term_3 = np.sin(x)  # array

    return float(np.sum((term_1 + term_2 + term_3) ** 2))


# ---------------------------------------------------------------------------
# 3. extended_rosenbrock
# ---------------------------------------------------------------------------


def extended_rosenbrock(x: np.ndarray) -> float:
    """Extended Rosenbrock function (pairwise variant).

    Parameters
    ----------
    x : np.ndarray
        Input vector of even length.

    Returns
    -------
    float
        Sum of Rosenbrock terms over consecutive pairs ``(x_p, x_d)``.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]
    c = 100.0

    term_1 = c * (x_d - x_p**2) ** 2
    term_2 = (1.0 - x_p) ** 2

    return float(np.sum(term_1 + term_2))


# ---------------------------------------------------------------------------
# 4. generalized_rosenbrock
# ---------------------------------------------------------------------------


def generalized_rosenbrock(x: np.ndarray) -> float:
    """Generalized Rosenbrock function (chain variant).

    Parameters
    ----------
    x : np.ndarray
        Input vector of arbitrary length ``n >= 2``.

    Returns
    -------
    float
        Sum of Rosenbrock terms along the chain ``x[i] -> x[i+1]``.
    """
    c = 100.0

    term_1 = c * (x[1:] - x[:-1] ** 2) ** 2
    term_2 = (1.0 - x[:-1]) ** 2

    return float(np.sum(term_1 + term_2))


# ---------------------------------------------------------------------------
# 5. extended_white_and_holst
# ---------------------------------------------------------------------------


def extended_white_and_holst(x: np.ndarray) -> float:
    """Extended White & Holst function (pairwise variant).

    Parameters
    ----------
    x : np.ndarray
        Input vector of even length.

    Returns
    -------
    float
        Sum of White & Holst terms over pairs ``(x_p, x_d)``.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]
    c = 100.0

    term_1 = c * (x_d - x_p**3) ** 2
    term_2 = (1.0 - x_p) ** 2

    return float(np.sum(term_1 + term_2))


# ---------------------------------------------------------------------------
# 6. extended_baele
# ---------------------------------------------------------------------------


def extended_baele(x: np.ndarray) -> float:
    """Extended Baele function.

    Parameters
    ----------
    x : np.ndarray
        Input vector of even length.

    Returns
    -------
    float
        Sum of three squared residual terms over pairs ``(x_p, x_d)``.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]

    term_1 = (1.5 - x_p * (1.0 - x_d)) ** 2
    term_2 = (2.25 - x_p * (1.0 - x_d**2)) ** 2
    term_3 = (2.625 - x_p * (1.0 - x_d**3)) ** 2

    return float(np.sum(term_1 + term_2 + term_3))


# ---------------------------------------------------------------------------
# 7. extended_penalty
# ---------------------------------------------------------------------------


def extended_penalty(x: np.ndarray) -> float:
    """Extended Penalty function.

    Parameters
    ----------
    x : np.ndarray
        Input vector of arbitrary length.

    Returns
    -------
    float
        Penalty value combining element-wise and global constraint violations.
    """
    term_1 = (x[:-1] - 1.0) ** 2
    term_2 = (np.sum(x**2) - 0.25) ** 2

    return float(np.sum(term_1) + term_2)


# ---------------------------------------------------------------------------
# 8. extended_himmelblau
# ---------------------------------------------------------------------------


def extended_himmelblau(x: np.ndarray) -> float:
    """Extended Himmelblau function.

    Parameters
    ----------
    x : np.ndarray
        Input vector of even length.

    Returns
    -------
    float
        Sum of Himmelblau terms over pairs ``(x_p, x_d)``.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]

    term_1 = (x_p**2 + x_d - 11.0) ** 2
    term_2 = (x_p + x_d**2 - 7.0) ** 2

    return float(np.sum(term_1 + term_2))


# ---------------------------------------------------------------------------
# 9. generalized_white_and_holst
# ---------------------------------------------------------------------------


def generalized_white_and_holst(x: np.ndarray) -> float:
    """Generalized White & Holst function (chain variant).

    Parameters
    ----------
    x : np.ndarray
        Input vector of arbitrary length ``n >= 2``.

    Returns
    -------
    float
        Sum of White & Holst terms along the chain ``x[i] -> x[i+1]``.
    """
    c = 100.0

    term_1 = c * (x[1:] - x[:-1] ** 3) ** 2
    term_2 = (1.0 - x[:-1]) ** 2

    return float(np.sum(term_1 + term_2))


# ---------------------------------------------------------------------------
# 10. extended_psc1
# ---------------------------------------------------------------------------


def extended_psc1(x: np.ndarray) -> float:
    """Extended PSC1 function.

    Parameters
    ----------
    x : np.ndarray
        Input vector of even length.

    Returns
    -------
    float
        Sum of polynomial and trigonometric terms over pairs ``(x_p, x_d)``.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]

    term_1 = (x_p**2 + x_d**2 + x_p * x_d) ** 2
    term_2 = np.sin(x_p) ** 2
    term_3 = np.cos(x_d) ** 2

    return float(np.sum(term_1 + term_2 + term_3))


# ---------------------------------------------------------------------------
# 11. extended_bd1
# ---------------------------------------------------------------------------


def extended_bd1(x: np.ndarray) -> float:
    """Extended BD1 function.

    Parameters
    ----------
    x : np.ndarray
        Input vector of even length.

    Returns
    -------
    float
        Sum of polynomial and exponential residual terms over pairs.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]

    term_1 = (x_p**2 + x_d - 2.0) ** 2
    term_2 = (np.exp(x_p - 1.0) - x_p) ** 2

    return float(np.sum(term_1 + term_2))


# ---------------------------------------------------------------------------
# 12. extended_maratos
# ---------------------------------------------------------------------------


def extended_maratos(x: np.ndarray) -> float:
    """Extended Maratos function.

    Parameters
    ----------
    x : np.ndarray
        Input vector of even length.

    Returns
    -------
    float
        Sum of linear and quadratic penalty terms over pairs.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]
    c = 100.0

    term_1 = x_p
    term_2 = c * (x_p**2 + x_d**2 - 1.0) ** 2

    return float(np.sum(term_1 + term_2))


# ---------------------------------------------------------------------------
# 13. extended_cliff
# ---------------------------------------------------------------------------


def extended_cliff(x: np.ndarray) -> float:
    """Extended Cliff function.

    Parameters
    ----------
    x : np.ndarray
        Input vector of even length.

    Returns
    -------
    float
        Sum of quadratic, linear, and exponential cliff terms over pairs.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]

    term_1 = ((x_p - 3.0) / 100.0) ** 2
    term_2 = x_p - x_d
    # Clip the exponent to prevent overflow in exp(20 * diff)
    cliff_exp = np.exp(np.clip(20.0 * (x_p - x_d), -100.0, 100.0))

    return float(np.sum(term_1 + term_2 + cliff_exp))


# ---------------------------------------------------------------------------
# 14. extended_hiebert
# ---------------------------------------------------------------------------


def extended_hiebert(x: np.ndarray) -> float:
    """Extended Hiebert function.

    Parameters
    ----------
    x : np.ndarray
        Input vector of even length.

    Returns
    -------
    float
        Sum of large-scale quadratic residual terms over pairs.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]

    term_1 = (x_p - 10.0) ** 2
    term_2 = (x_p * x_d - 50000.0) ** 2

    return float(np.sum(term_1 + term_2))


# ---------------------------------------------------------------------------
# 15. extended_tridiagonal_1
# ---------------------------------------------------------------------------


def extended_tridiagonal_1(x: np.ndarray) -> float:
    """Extended Tridiagonal 1 function.

    Parameters
    ----------
    x : np.ndarray
        Input vector of even length.

    Returns
    -------
    float
        Sum of quadratic and quartic terms over pairs ``(x_p, x_d)``.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]

    term_1 = (x_p + x_d - 3.0) ** 2
    term_2 = (x_p - x_d + 1.0) ** 4

    return float(np.sum(term_1 + term_2))


# ---------------------------------------------------------------------------
# 16. extended_tridiagonal_2
# ---------------------------------------------------------------------------


def extended_tridiagonal_2(x: np.ndarray) -> float:
    """Extended Tridiagonal 2 function (chain variant).

    Parameters
    ----------
    x : np.ndarray
        Input vector of arbitrary length ``n >= 2``.

    Returns
    -------
    float
        Sum of product-based quadratic terms along the chain.
    """
    c = 0.1

    term_1 = (x[1:] * x[:-1] - 1.0) ** 2
    term_2 = c * (x[:-1] + 1.0) ** 2

    return float(np.sum(term_1 + term_2))


# ---------------------------------------------------------------------------
# 17. extended_denschnb
# ---------------------------------------------------------------------------


def extended_denschnb(x: np.ndarray) -> float:
    """Extended Denschnb function.

    Parameters
    ----------
    x : np.ndarray
        Input vector of even length.

    Returns
    -------
    float
        Sum of quadratic and mixed terms over pairs ``(x_p, x_d)``.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]

    term_1 = (x_p - 2.0) ** 2
    term_2 = term_1 * x_d**2
    term_3 = (x_d + 1.0) ** 2

    return float(np.sum(term_1 + term_2 + term_3))


# ---------------------------------------------------------------------------
# 18. extended_denschnf
# ---------------------------------------------------------------------------


def extended_denschnf(x: np.ndarray) -> float:
    """Extended Denschnf function.

    Parameters
    ----------
    x : np.ndarray
        Input vector of even length.

    Returns
    -------
    float
        Sum of squared quadratic combinations and a polynomial penalty over pairs.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]

    term_1 = 2.0 * (x_p + x_d) ** 2
    term_2 = (x_p - x_d) ** 2 - 8.0
    term_3 = (5.0 * x_p**2 + (x_p - 3.0) ** 2 - 9.0) ** 2

    return float(np.sum((term_1 + term_2) ** 2 + term_3))


# ---------------------------------------------------------------------------
# 19. extended_quadratic_exponential_ep1
# ---------------------------------------------------------------------------


def extended_quadratic_exponential_ep1(x: np.ndarray) -> float:
    """Extended Quadratic Exponential EP1 function.

    Parameters
    ----------
    x : np.ndarray
        Input vector of even length.

    Returns
    -------
    float
        Sum of exponential, quadratic, and shifted quadratic terms over pairs.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]

    diff = x_p - x_d

    term_1 = (np.exp(diff) - 5.0) ** 2
    term_2 = diff**2
    term_3 = (diff - 11.0) ** 2

    return float(np.sum(term_1 + term_2 * term_3))
