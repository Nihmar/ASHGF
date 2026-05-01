"""
Classic optimisation test functions.

This module provides a collection of well-known analytical benchmark
functions commonly used to evaluate optimisation algorithms.  Each
function accepts a 1-D :class:`numpy.ndarray` and returns a scalar
:class:`float` (with the exception of ``relu`` and ``softmax``, which
are activation primitives returning arrays).

All functions are implemented as standalone routines (no classes) with
full NumPy-style docstrings and type hints.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Classical benchmark functions
# ---------------------------------------------------------------------------


def sphere(x: np.ndarray) -> float:
    r"""Sphere function.

    .. math::

        f(\mathbf{x}) = \mathbf{x}^\top \mathbf{x}
                      = \sum_{i=1}^{n} x_i^2

    Global minimum: :math:`f(\mathbf{0}) = 0`.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    return float(np.dot(x, x))


def rastrigin(x: np.ndarray) -> float:
    r"""Rastrigin function.

    .. math::

        f(\mathbf{x}) = 10n + \sum_{i=1}^{n}
                        \bigl[x_i^2 - 10\cos(2\pi x_i)\bigr]

    Global minimum: :math:`f(\mathbf{0}) = 0`.

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
    return float(10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * math.pi * x)))


def ackley(x: np.ndarray) -> float:
    r"""Ackley function.

    .. math::

        f(\mathbf{x}) = -a \exp\!\bigl(-b \sqrt{\tfrac{1}{n}
                        \sum_{i=1}^{n} x_i^2}\bigr)
                        - \exp\!\bigl(\tfrac{1}{n}
                        \sum_{i=1}^{n} \cos(c x_i)\bigr)
                        + a + e

    with :math:`a = 20`, :math:`b = 0.2`, :math:`c = 2\pi`.

    Global minimum: :math:`f(\mathbf{0}) = 0`.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    a: float = 20.0
    b: float = 0.2
    c: float = 2.0 * math.pi

    n = len(x)
    term_1: float = -a * np.exp(-b * np.sqrt(np.mean(x**2)))
    term_2: float = -np.exp(np.mean(np.cos(c * x)))
    term_3: float = a + math.e

    return float(term_1 + term_2 + term_3)


def griewank(x: np.ndarray) -> float:
    r"""Griewank function.

    .. math::

        f(\mathbf{x}) = \frac{1}{4000}\sum_{i=1}^{n} x_i^2
                        - \prod_{i=1}^{n} \cos\!\Bigl(
                          \frac{x_i}{\sqrt{i}}\Bigr) + 1

    Global minimum: :math:`f(\mathbf{0}) = 0`.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n: int = len(x)
    term_1: float = (1.0 / 4000.0) * np.sum(x**2)
    term_2: float = float(np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1)))))
    return float(term_1 - term_2 + 1.0)


def levy(x: np.ndarray) -> float:
    r"""Levy function.

    .. math::

        w_i &= 1 + \frac{x_i - 1}{4}

        f(\mathbf{x}) &= \sin^2(\pi w_1) \\
        &\quad + \sum_{i=1}^{n-1} (w_i - 1)^2
           \bigl[1 + 10\sin^2(\pi w_i + 1)\bigr] \\
        &\quad + (w_n - 1)^2 \bigl[1 + \sin^2(2\pi w_n)\bigr]

    Global minimum: :math:`f(\mathbf{1}) = 0`.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    w: np.ndarray = 1.0 + (x - 1.0) / 4.0
    w_first: float = float(w[0])
    w_last: float = float(w[-1])
    w_interior: np.ndarray = w[:-1]

    term_1: float = float(np.sin(math.pi * w_first) ** 2)
    term_2: float = float(
        np.sum(
            (w_interior - 1.0) ** 2
            * (1.0 + 10.0 * (np.sin(math.pi * w_interior + 1.0) ** 2))
        )
    )
    term_3: float = float(
        (w_last - 1.0) ** 2 * (1.0 + np.sin(2.0 * math.pi * w_last) ** 2)
    )

    return float(term_1 + term_2 + term_3)


def schwefel(x: np.ndarray) -> float:
    r"""Schwefel function.

    .. math::

        f(\mathbf{x}) = 418.9829\,n
                        - \sum_{i=1}^{n} x_i
                          \sin\!\bigl(\sqrt{|x_i|}\bigr)

    Global minimum: :math:`f(\mathbf{x}^*) \approx 0` at
    :math:`x_i^* = 420.9687` for all :math:`i`.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n: int = len(x)
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def sum_of_different_powers(x: np.ndarray) -> float:
    r"""Sum of different powers function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} |x_i|^{i+1}

    Global minimum: :math:`f(\mathbf{0}) = 0`.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n: int = len(x)
    exponents: np.ndarray = np.arange(2, n + 2)
    return float(np.sum(np.abs(x) ** exponents))


def trid(x: np.ndarray) -> float:
    r"""Tridiagonal (Trid) function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} (x_i - 1)^2
                        + \sum_{i=2}^{n} x_i x_{i-1}

    Global minimum depends on :math:`n`.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    term_1: float = float(np.sum((x - 1.0) ** 2))
    term_2: float = float(np.sum(x[1:] * x[:-1]))
    return float(term_1 + term_2)


def zakharov(x: np.ndarray) -> float:
    r"""Zakharov function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n} x_i^2
                        + \Bigl(\tfrac{1}{2}\sum_{i=1}^{n} i x_i\Bigr)^2
                        + \Bigl(\tfrac{1}{2}\sum_{i=1}^{n} i x_i\Bigr)^4

    Global minimum: :math:`f(\mathbf{0}) = 0`.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    n: int = len(x)
    i = np.arange(1, n + 1)
    half_sum = 0.5 * np.sum(i * x)
    term_1: float = float(np.sum(x**2))
    term_2: float = float(half_sum**2)
    term_3: float = float(term_2**2)
    return float(term_1 + term_2 + term_3)


def cosine(x: np.ndarray) -> float:
    r"""Cosine mixture function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-1}
                        \cos\!\bigl(-0.5 x_{i+1} + x_i^2\bigr)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    return float(np.sum(np.cos(-0.5 * x[1:] + x[:-1] ** 2)))


def sine(x: np.ndarray) -> float:
    r"""Sine mixture function.

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-1}
                        \sin\!\bigl(-0.5 x_{i+1} + x_i^2\bigr)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    float
        Function value.
    """
    return float(np.sum(np.sin(-0.5 * x[1:] + x[:-1] ** 2)))


def sincos(x: np.ndarray) -> float:
    r"""Sine-Cosine mixture function.

    Evaluates even- and odd-indexed sub-vectors:

    .. math::

        \mathbf{x}_p &= [x_0, x_2, x_4, \dots] \\
        \mathbf{x}_d &= [x_1, x_3, x_5, \dots] \\
        f(\mathbf{x}) &= \sum
                        \bigl[
                            (x_{p,j}^2 + x_{d,j}^2 + x_{p,j}x_{d,j})^2
                            + \sin^2(x_{p,j})
                            + \cos^2(x_{d,j})
                        \bigr]

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.  When ``n`` is odd the last
        element of ``x[::2]`` is ignored because ``x[1::2]`` will be
        one element shorter.

    Returns
    -------
    float
        Function value.
    """
    x_p: np.ndarray = x[::2]
    x_d: np.ndarray = x[1::2]

    # Truncate to equal length if needed (odd n).
    m: int = min(len(x_p), len(x_d))
    x_p = x_p[:m]
    x_d = x_d[:m]

    term_1: np.ndarray = (x_p**2 + x_d**2 + x_p * x_d) ** 2
    term_2: np.ndarray = np.sin(x_p) ** 2
    term_3: np.ndarray = np.cos(x_d) ** 2

    return float(np.sum(term_1 + term_2 + term_3))


# ---------------------------------------------------------------------------
# Activation-style primitives (return arrays)
# ---------------------------------------------------------------------------


def relu(x: np.ndarray) -> np.ndarray:
    r"""Rectified Linear Unit (ReLU) activation.

    .. math::

        \mathrm{ReLU}(x_i) = \max(0, x_i)

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Array of the same shape as ``x`` with the ReLU activation applied.
    """
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    r"""Softmax function.

    .. math::

        \mathrm{softmax}(x_i) =
        \frac{e^{x_i - \max(\mathbf{x})}}
             {\sum_{j} e^{x_j - \max(\mathbf{x})}}

    The subtraction of the maximum improves numerical stability
    and must be applied to both the numerator and the denominator.

    Parameters
    ----------
    x : np.ndarray
        Input vector of shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Softmax probabilities (same shape as ``x``, summing to 1).
    """
    y: np.ndarray = np.exp(x - np.max(x))
    return y / np.sum(y)
