"""Benchmark optimisation test functions.

This module provides 48 standalone functions commonly used to evaluate
optimisation algorithms.  Each function accepts a 1-D :class:`numpy.ndarray`
and returns a scalar :class:`float`.

All functions are implemented as standalone routines (no classes) with
NumPy-style docstrings and type hints.  They have been converted from an
older class-based implementation by replacing ``self.`` references and
``self.power(x)`` calls with direct vectorised NumPy operations.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def perturbed_quadratic(x: np.ndarray) -> float:
    r"""Perturbed Quadratic function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} i x_i^2
                        + \frac{1}{100}\Bigl(\sum_{i=1}^{n} x_i\Bigr)^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    i = np.arange(1, n + 1)
    term_1 = float(np.sum(i * x**2))
    term_2 = (1.0 / 100.0) * float(np.sum(x)) ** 2
    return float(term_1 + term_2)


def raydan_1(x: np.ndarray) -> float:
    r"""Raydan 1 function.

    .. math::

        f(\mathbf{x}) = \frac{1}{10}\sum_{i=1}^{n}
                        i\,(e^{x_i} - x_i)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    i = np.arange(1, n + 1)
    return float((1.0 / 10.0) * np.sum(i * (np.exp(x) - x)))


def raydan_2(x: np.ndarray) -> float:
    r"""Raydan 2 function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} (e^{x_i} - x_i)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    return float(np.sum(np.exp(x) - x))


def diagonal_1(x: np.ndarray) -> float:
    r"""Diagonal 1 function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} (e^{x_i} - i x_i)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.sum(np.exp(x) - i * x))


def diagonal_2(x: np.ndarray) -> float:
    r"""Diagonal 2 function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} \Bigl(e^{x_i} - \frac{x_i}{i}\Bigr)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.sum(np.exp(x) - x / i))


def diagonal_3(x: np.ndarray) -> float:
    r"""Diagonal 3 function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} (e^{x_i} - i \sin x_i)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.sum(np.exp(x) - i * np.sin(x)))


def hager(x: np.ndarray) -> float:
    r"""Hager function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} (e^{x_i} - \sqrt{i}\, x_i)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.sum(np.exp(x) - np.sqrt(i) * x))


def generalized_tridiagonal_1(x: np.ndarray) -> float:
    r"""Generalized Tridiagonal 1 function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-1} \bigl[
            (x_i + x_{i+1} - 3)^2 + (x_i - x_{i+1} + 1)^4
        \bigr]

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = (x[:-1] + x[1:] - 3.0) ** 2
    term_2 = (x[:-1] - x[1:] + 1.0) ** 4
    return float(np.sum(term_1 + term_2))


def diagonal_4(x: np.ndarray) -> float:
    r"""Diagonal 4 function.

    .. math::

        f(\mathbf{x}) = \frac{1}{2}\Bigl(
            \sum_{j} x_{p,j}^2 + c \sum_{j} x_{d,j}^2\Bigr)
        \quad\text{with } c = 100

    where :math:`x_p = [x_0, x_2, \dots]` and
    :math:`x_d = [x_1, x_3, \dots]`.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]
    c = 100.0
    return float(0.5 * (np.sum(x_p**2) + c * np.sum(x_d**2)))


def diagonal_5(x: np.ndarray) -> float:
    r"""Diagonal 5 function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n}
                        \log(e^{x_i} + e^{-x_i})

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    return float(np.sum(np.log(np.exp(x) + np.exp(-x))))


def diagonal_7(x: np.ndarray) -> float:
    r"""Diagonal 7 function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n}
                        (e^{x_i} - 2x_i - x_i^2)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    return float(np.sum(np.exp(x) - 2.0 * x - x**2))


def diagonal_8(x: np.ndarray) -> float:
    r"""Diagonal 8 function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n}
                        (x_i e^{x_i} - 2x_i - x_i^2)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    return float(np.sum(x * np.exp(x) - 2.0 * x - x**2))


def diagonal_9(x: np.ndarray) -> float:
    r"""Diagonal 9 function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} (e^{x_i} - i x_i)
                        + 10000\, x_n^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    i = np.arange(1, n + 1)
    term_1 = float(np.sum(np.exp(x) - i * x))
    term_2 = 10000.0 * float(x[-1] ** 2)
    return float(term_1 + term_2)


def fletcbv3(x: np.ndarray) -> float:
    r"""Fletcher boundary value problem 3 (FLETCBV3).

    .. math::

        f(\mathbf{x}) = \frac{p}{2}(x_1^2 + x_n^2)
            + \frac{p}{2}\sum_{i=1}^{n-1}(x_i - x_{i+1})^2
            + \sum_{i=1}^{n}\Bigl[
                \frac{p(h^2+2)}{h^2} x_i
                + \frac{c p}{h^2} \cos x_i
            \Bigr]

    with :math:`p = 10^{-8}`, :math:`h = 1/(n+1)`, :math:`c = 1`.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    p = 1e-8
    h = 1.0 / (n + 1)
    c = 1.0
    h_sq = h**2

    term_1 = 0.5 * p * (x[0] ** 2 + x[-1] ** 2)
    term_2 = np.sum((p / 2.0) * (x[:-1] - x[1:]) ** 2)
    coeff1 = p * (h_sq + 2.0) / h_sq
    coeff2 = c * p / h_sq
    term_3 = np.sum(coeff1 * x + coeff2 * np.cos(x))

    return float(term_1 + term_2 + term_3)


def fletchcr(x: np.ndarray) -> float:
    r"""Fletcher chained Rosenbrock-like (FLETCHCR) function.

    .. math::

        f(\mathbf{x}) = c \sum_{i=1}^{n-1}
            \bigl(x_{i+1} - x_i + 1 - x_i^2\bigr)^2
        \quad\text{with } c = 100

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    c = 100.0
    return float(np.sum(c * (x[1:] - x[:-1] + 1.0 - x[:-1] ** 2) ** 2))


def bdqrtic(x: np.ndarray) -> float:
    r"""BDQRTIC function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-3}
            \bigl[(-4x_i + 3)^2 + \bigl(
                x_i^2 + 2x_{i+1}^2 + 3x_{i+2}^2 + 4x_{i+3}^2
                + 5x_n\bigr)^2\bigr]

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 4``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = (-4.0 * x[:-3] + 3.0) ** 2
    x_sq = x**2
    term_2 = (
        x_sq[:-3] + 2.0 * x_sq[1:-2] + 3.0 * x_sq[2:-1] + 4.0 * x_sq[3:] + 5.0 * x[-1]
    ) ** 2
    return float(np.sum(term_1 + term_2))


def tridia(x: np.ndarray) -> float:
    r"""TRIDIA function.

    .. math::

        f(\mathbf{x}) = \gamma (\delta x_1 - 1)^2
            + \sum_{i=2}^{n} i\,
                (\alpha x_i - \beta x_{i-1})^2

    with :math:`\alpha=2`, :math:`\beta=1`, :math:`\gamma=1`,
    :math:`\delta=1`.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    alfa = 2.0
    beta = 1.0
    gamma = 1.0
    delta = 1.0

    term_1 = gamma * (delta * x[0] - 1.0) ** 2
    term_2 = np.sum(np.arange(2, n + 1) * (alfa * x[1:] - beta * x[:-1]) ** 2)
    return float(term_1 + term_2)


def arwhead(x: np.ndarray) -> float:
    r"""ARWHEAD function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-1} (-4x_i + 3)
            + \sum_{i=1}^{n-1} (x_i^2 + x_n^2)^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = float(np.sum(-4.0 * x[:-1] + 3.0))
    x_sq = x**2
    term_2 = float(np.sum((x_sq[:-1] + x_sq[-1]) ** 2))
    return float(term_1 + term_2)


def nondia(x: np.ndarray) -> float:
    r"""NONDIA function.

    .. math::

        f(\mathbf{x}) = (x_1 - 1)^2
            + 100 \sum_{i=2}^{n} (x_1 - x_i^2)^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = (x[0] - 1.0) ** 2
    term_2 = 100.0 * np.sum((x[0] - x[1:] ** 2) ** 2)
    return float(term_1 + term_2)


def nondquar(x: np.ndarray) -> float:
    r"""NONDQUAR function.

    .. math::

        f(\mathbf{x}) = (x_1 - x_2)^2
            + \sum_{i=1}^{n-2} (x_i + x_{i+1} + x_n)^4
            + (x_{n-1} + x_n)^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 3``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = (x[0] - x[1]) ** 2
    term_2 = np.sum((x[:-2] + x[1:-1] + x[-1]) ** 4)
    term_3 = (x[-2] + x[-1]) ** 2
    return float(term_1 + term_2 + term_3)


def dqdrtic(x: np.ndarray) -> float:
    r"""DQDRTIC function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-2}
            (x_i^2 + c x_{i+1}^2 + d x_{i+2}^2)
        \quad\text{with } c = 100,\; d = 100

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 3``.

    Returns
    -------
    float
        Function value.
    """
    x_sq = x**2
    c = 100.0
    d = 100.0
    return float(np.sum(x_sq[:-2] + c * x_sq[1:-1] + d * x_sq[2:]))


def eg2(x: np.ndarray) -> float:
    r"""EG2 function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-1}
            \sin(x_1 + x_i^2 - 1)
            + \frac{1}{2} \sin(x_n^2)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    x_sq = x**2
    term_1 = float(np.sum(np.sin(x[0] + x_sq[:-1] - 1.0)))
    term_2 = 0.5 * np.sin(x_sq[-1])
    return float(term_1 + term_2)


def broyden_tridiagonal(x: np.ndarray) -> float:
    r"""Broyden Tridiagonal function.

    .. math::

        f(\mathbf{x}) = (3x_1 - 2x_1^2)^2
            + \sum_{i=2}^{n-1}
                (3x_i - 2x_i^2 - x_{i-1} - 2x_{i+1} + 1)^2
            + (3x_n - 2x_n^2 - x_{n-1} + 1)^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 3``.

    Returns
    -------
    float
        Function value.
    """
    x_sq = x**2

    term_1 = (3.0 * x[0] - 2.0 * x_sq[0]) ** 2
    term_2 = np.sum(
        (3.0 * x[1:-1] - 2.0 * x_sq[1:-1] - x[:-2] - 2.0 * x[2:] + 1.0) ** 2
    )
    term_3 = (3.0 * x[-1] - 2.0 * x_sq[-1] - x[-2] + 1.0) ** 2

    return float(term_1 + term_2 + term_3)


def almost_perturbed_quadratic(x: np.ndarray) -> float:
    r"""Almost Perturbed Quadratic function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} i x_i^2
            + \frac{1}{100}(x_1 + x_n)^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    i = np.arange(1, n + 1)
    term_1 = float(np.sum(i * x**2))
    term_2 = (1.0 / 100.0) * (x[0] + x[-1]) ** 2
    return float(term_1 + term_2)


def liarwhd(x: np.ndarray) -> float:
    r"""LIARWHD function (corrected version).

    .. math::

        f(\mathbf{x}) = 4\sum_{i=1}^{n} (x_i^2 - x_1)^2
            + \sum_{i=1}^{n} (x_i - 1)^2

    .. note::
        This is the corrected version that replaces the earlier buggy
        implementation.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    # Corrected version: replaced the earlier buggy implementation.
    term_1 = 4.0 * np.sum((x**2 - x[0]) ** 2)
    term_2 = np.sum((x - 1.0) ** 2)
    return float(term_1 + term_2)


def power(x: np.ndarray) -> float:
    r"""Power (weighted sum of squares) function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} i x_i^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.sum(i * x**2))


def engval1(x: np.ndarray) -> float:
    r"""ENGVAL1 function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-1}
            (x_i^2 + x_{i+1}^2)^2
            + \sum_{i=1}^{n-1} (-4x_i + 3)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    x_sq = x**2
    term_1 = float(np.sum((x_sq[:-1] + x_sq[1:]) ** 2))
    term_2 = float(np.sum(-4.0 * x[:-1] + 3.0))
    return float(term_1 + term_2)


def edensch(x: np.ndarray) -> float:
    r"""EDENSCH function.

    .. math::

        f(\mathbf{x}) = 16
            + \sum_{i=1}^{n-1} \bigl[
                (x_i - 2)^4
                + (x_i x_{i+1} + 2x_{i+1})^2
                + (x_{i+1} + 1)^2
            \bigr]

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = (x[:-1] - 2.0) ** 4
    term_2 = (x[:-1] * x[1:] + 2.0 * x[1:]) ** 2
    term_3 = (x[1:] + 1.0) ** 2
    return float(16.0 + np.sum(term_1 + term_2 + term_3))


def indef(x: np.ndarray) -> float:
    r"""INDEF function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} x_i
            + \frac{1}{2}\sum_{i=2}^{n-1}
                \cos(2x_i - x_n - x_1)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 3``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = float(np.sum(x))
    term_2 = 0.5 * float(np.sum(np.cos(2.0 * x[1:-1] - x[-1] - x[0])))
    return float(term_1 + term_2)


def cube(x: np.ndarray) -> float:
    r"""CUBE function.

    .. math::

        f(\mathbf{x}) = (x_1 - 1)^2
            + 100 \sum_{i=2}^{n} (x_i - x_{i-1}^3)^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = (x[0] - 1.0) ** 2
    term_2 = 100.0 * np.sum((x[1:] - x[:-1] ** 3) ** 2)
    return float(term_1 + term_2)


def bdexp(x: np.ndarray) -> float:
    r"""BDEXP function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-2}
            (x_i + x_{i+1})
            \exp\!\bigl(-x_{i+2}(x_i + x_{i+1})\bigr)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 3``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = x[:-2] + x[1:-1]
    term_2 = np.exp(-x[2:] * term_1)
    return float(np.sum(term_1 * term_2))


def genhumps(x: np.ndarray) -> float:
    r"""GENHUMPS function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-1} \bigl[
            \sin^2(2 x_i)\,\sin^2(2 x_{i+1})
            + 0.05(x_i^2 + x_{i+1}^2)
        \bigr]

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = np.sin(2.0 * x[:-1]) ** 2
    term_2 = np.sin(2.0 * x[1:]) ** 2
    term_3 = 0.05 * (x[:-1] ** 2 + x[1:] ** 2)
    return float(np.sum(term_1 * term_2 + term_3))


def mccormck(x: np.ndarray) -> float:
    r"""McCormick function (chained variant).

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-1} \bigl[
            -1.5 x_i + 2.5 x_{i+1} + 1
            + (x_i - x_{i+1})^2
            + \sin(x_i + x_{i+1})
        \bigr]

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = -1.5 * x[:-1] + 2.5 * x[1:] + 1.0
    term_2 = (x[:-1] - x[1:]) ** 2
    term_3 = np.sin(x[:-1] + x[1:])
    return float(np.sum(term_1 + term_2 + term_3))


def nonscomp(x: np.ndarray) -> float:
    r"""NONSCOMP function.

    .. math::

        f(\mathbf{x}) = (x_1 - 1)^2
            + 4 \sum_{i=2}^{n} (x_i - x_{i-1}^2)^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = (x[0] - 1.0) ** 2
    term_2 = 4.0 * np.sum((x[1:] - x[:-1] ** 2) ** 2)
    return float(term_1 + term_2)


def vardim(x: np.ndarray) -> float:
    r"""VARDIM function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} (x_i - 1)^2
            + \Bigl(\sum_{i=1}^{n} i x_i
                  - \frac{n(n+1)}{2}\Bigr)^2
            + \Bigl(\sum_{i=1}^{n} i x_i
                  - \frac{n(n+1)}{2}\Bigr)^4

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    term_1 = float(np.sum((x - 1.0) ** 2))
    weighted_sum = float(np.sum(np.arange(1, n + 1) * x))
    offset = float(n * (n + 1) / 2.0)
    term_2 = (weighted_sum - offset) ** 2
    term_3 = term_2**2
    return float(term_1 + term_2 + term_3)


def quartc(x: np.ndarray) -> float:
    r"""QUARTC function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} (x_i - 1)^4

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    return float(np.sum((x - 1.0) ** 4))


def sinquad(x: np.ndarray) -> float:
    r"""SINQUAD function.

    .. math::

        f(\mathbf{x}) = (x_1 - 1)^4
            + \sum_{i=2}^{n-1}
                \bigl(\sin(x_i - x_n) - x_1^2 + x_i^2\bigr)^2
            + (x_n^2 - x_1^2)^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 3``.

    Returns
    -------
    float
        Function value.
    """
    x_sq = x**2
    term_1 = (x[0] - 1.0) ** 4
    term_2 = np.sum((np.sin(x[1:-1] - x[-1]) - x_sq[0] + x_sq[1:-1]) ** 2)
    term_3 = (x_sq[-1] - x_sq[0]) ** 2
    return float(term_1 + term_2 + term_3)


def dixon3dq(x: np.ndarray) -> float:
    r"""DIXON3DQ function.

    .. math::

        f(\mathbf{x}) = (x_1 - 1)^2
            + \sum_{i=1}^{n-1} (x_i - x_{i+1})^2
            + (x_n - 1)^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = (x[0] - 1.0) ** 2
    term_2 = float(np.sum((x[:-1] - x[1:]) ** 2))
    term_3 = (x[-1] - 1.0) ** 2
    return float(term_1 + term_2 + term_3)


def biggsb1(x: np.ndarray) -> float:
    r"""BIGGSB1 function.

    .. math::

        f(\mathbf{x}) = (x_1 - 1)^2
            + \sum_{i=2}^{n} (x_i - x_{i-1})^2
            + (x_n - 1)^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = (x[0] - 1.0) ** 2
    term_2 = float(np.sum((x[1:] - x[:-1]) ** 2))
    term_3 = (x[-1] - 1.0) ** 2
    return float(term_1 + term_2 + term_3)


def generalized_quartic(x: np.ndarray) -> float:
    r"""Generalized Quartic function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-1}
            \bigl[x_i^2 + (x_{i+1} + x_i^2)^2\bigr]

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = x[:-1] ** 2
    term_2 = (x[1:] + term_1) ** 2
    return float(np.sum(term_1 + term_2))


def fh3(x: np.ndarray) -> float:
    r"""FH3 function.

    .. math::

        f(\mathbf{x}) = \Bigl(\sum_{i=1}^{n} x_i\Bigr)^2
            + \sum_{i=1}^{n} (x_i e^{x_i} - 2x_i - x_i^2)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = float(np.sum(x)) ** 2
    term_2 = float(np.sum(x * np.exp(x) - 2.0 * x - x**2))
    return float(term_1 + term_2)


def himmelbg(x: np.ndarray) -> float:
    r"""HIMMELBG function.

    .. math::

        f(\mathbf{x}) = \sum_{j}
            (2 x_{p,j}^2 + 3 x_{d,j}^2)
            \exp(-x_{p,j} - x_{d,j})

    where :math:`x_p = [x_0, x_2, \dots]` and
    :math:`x_d = [x_1, x_3, \dots]`.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]
    # Truncate to equal length if needed (odd n).
    m = min(len(x_p), len(x_d))
    x_p = x_p[:m]
    x_d = x_d[:m]

    term_1 = 2.0 * x_p**2 + 3.0 * x_d**2
    term_2 = np.exp(-x_p - x_d)
    return float(np.sum(term_1 * term_2))


def himmelh(x: np.ndarray) -> float:
    r"""HIMMELH function.

    .. math::

        f(\mathbf{x}) = \sum_{j}
            \bigl(-3 x_{p,j} - 2 x_{d,j} + 2
                  + x_{p,j}^3 + x_{d,j}^2\bigr)

    where :math:`x_p = [x_0, x_2, \dots]` and
    :math:`x_d = [x_1, x_3, \dots]`.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    x_p = x[:-1:2]
    x_d = x[1::2]
    # Truncate to equal length if needed (odd n).
    m = min(len(x_p), len(x_d))
    x_p = x_p[:m]
    x_d = x_d[:m]

    return float(np.sum(-3.0 * x_p - 2.0 * x_d + 2.0 + x_p**3 + x_d**2))


def quadratic_qf1(x: np.ndarray) -> float:
    r"""Quadratic QF1 function.

    .. math::

        f(\mathbf{x}) = \frac{1}{2}\sum_{i=1}^{n} i x_i^2
            + x_n

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    i = np.arange(1, n + 1)
    term_1 = 0.5 * float(np.sum(i * x**2))
    term_2 = float(x[-1])
    return float(term_1 + term_2)


def extended_quadratic_penalty_qp1(x: np.ndarray) -> float:
    r"""Extended Quadratic Penalty QP1 function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-1} (x_i^2 - 2)^2
            + \Bigl(\sum_{i=1}^{n} x_i^2 - 0.5\Bigr)^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = float(np.sum((x[:-1] ** 2 - 2.0) ** 2))
    term_2 = (float(np.sum(x**2)) - 0.5) ** 2
    return float(term_1 + term_2)


def extended_quadratic_penalty_qp2(x: np.ndarray) -> float:
    r"""Extended Quadratic Penalty QP2 function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-1}
            (x_i^2 - \sin x_i)^2
            + \Bigl(\sum_{i=1}^{n} x_i^2 - 100\Bigr)^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)`` with ``n >= 2``.

    Returns
    -------
    float
        Function value.
    """
    term_1 = float(np.sum((x[:-1] ** 2 - np.sin(x[:-1])) ** 2))
    term_2 = (float(np.sum(x**2)) - 100.0) ** 2
    return float(term_1 + term_2)


def quadratic_qf2(x: np.ndarray) -> float:
    r"""Quadratic QF2 function.

    .. math::

        f(\mathbf{x}) = \frac{1}{2}\sum_{i=1}^{n}
            i\,(x_i^2 - 1)^2 + x_n

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    i = np.arange(1, n + 1)
    term_1 = 0.5 * float(np.sum(i * (x**2 - 1.0) ** 2))
    term_2 = float(x[-1])
    return float(term_1 + term_2)


def perturbed_quadratic_diagonal(x: np.ndarray) -> float:
    r"""Perturbed Quadratic Diagonal function.

    .. math::

        f(\mathbf{x}) = \Bigl(\sum_{i=1}^{n} x_i\Bigr)^2
            + \sum_{i=1}^{n} \frac{i}{100}\, x_i^2

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    term_1 = float(np.sum(x)) ** 2
    term_2 = float(np.sum((np.arange(1, n + 1) / 100.0) * x**2))
    return float(term_1 + term_2)
