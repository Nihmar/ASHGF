"""Benchmark objective functions.

Every function in this module implements its formula exactly as given by the thesis
(``thesis/chapters/numericalexperiments.tex``, table "ListFunctions") whenever the thesis
lists it; otherwise it follows the frozen prototype in ``original_python_code/functions.py``.
Wherever the two sources disagree, the canonical name carries the thesis version and the
frozen-prototype version is kept under a ``*_original`` alias for parity studies (see
``LEGACY_VARIANTS``).

Conventions
-----------
- All callables are *batch-native*: they accept an array of shape ``( ..., dim )`` and return
  an array of shape ``( ... , )``. A single point of shape ``(dim,)`` therefore works out of
  the box.
- Pair-based ("extended") functions split the trailing axis into even-indexed entries
  ``x_p`` and odd-indexed entries ``x_d``; when the dimension is odd the last entry is
  dropped, matching the frozen prototype.
- No Python-level loops over data; only small structural loops over fixed pair sets.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Type alias: a vectorized objective taking points of shape (..., dim) -> (...,).
VectorizedFn = "callable[[NDArray[np.float64]], NDArray[np.float64]]"


def _pair_split(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split the trailing axis into even-indexed and odd-indexed entries.

    When the dimension is odd the final entry is dropped so that both halves have equal
    length (same behavior as the frozen prototype).
    """
    d = x.shape[-1]
    stop = d - 1 if d % 2 else d
    return x[..., :stop:2], x[..., 1:stop:2]


# ---------------------------------------------------------------------------
# Canonical implementations (thesis formulas where listed there)
# ---------------------------------------------------------------------------


def sphere(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i x_i^2."""
    return np.sum(x**2, axis=-1)


def power(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i i x_i^2."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    return np.sum(idx * x**2, axis=-1)


def perturbed_quadratic(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i i x_i^2 + (sum_i x_i)^2 / 100."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    return np.sum(idx * x**2, axis=-1) + (np.sum(x, axis=-1) ** 2) / 100.0


def almost_perturbed_quadratic(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i i x_i^2 + (x_1 + x_n)^2 / 100."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    return np.sum(idx * x**2, axis=-1) + ((x[..., 0] + x[..., -1]) ** 2) / 100.0


def perturbed_quadratic_diagonal(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = (sum_i x_i)^2 + sum_i (i/100) x_i^2."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    return np.sum(x, axis=-1) ** 2 + np.sum((idx / 100.0) * x**2, axis=-1)


def generalized_rosenbrock(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_{i=1}^{n-1} [c (x_{i+1} - x_i^2)^2 + (1 - x_i)^2], c = 100."""
    xp, xd = x[..., :-1], x[..., 1:]
    return np.sum(100.0 * (xd - xp**2) ** 2 + (1.0 - xp) ** 2, axis=-1)


def extended_rosenbrock(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_{i=1}^{n/2} [100 (x_{2i} - x_{2i-1}^2)^2 + (1 - x_{2i-1})^2]."""
    xp, xd = _pair_split(x)
    return np.sum(100.0 * (xd - xp**2) ** 2 + (1.0 - xp) ** 2, axis=-1)


def generalized_white_and_holst(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_{i=1}^{n-1} [c (x_{i+1} - x_i^3)^2 + (1 - x_i)^2], c = 100."""
    xp, xd = x[..., :-1], x[..., 1:]
    return np.sum(100.0 * (xd - xp**3) ** 2 + (1.0 - xp) ** 2, axis=-1)


def extended_white_and_holst(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_{i=1}^{n/2} [100 (x_{2i} - x_{2i-1}^3)^2 + (1 - x_{2i-1})^2]."""
    xp, xd = _pair_split(x)
    return np.sum(100.0 * (xd - xp**3) ** 2 + (1.0 - xp) ** 2, axis=-1)


def extended_trigonometric(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Thesis formula:

    F(x) = sum_{i=1}^{n} [(n - sum_j cos x_j) + i (1 - cos x_i) - sin x_i]^2

    Note: the frozen prototype adds ``sin`` here instead of subtracting it; that variant is
    available as ``extended_trigonometric_original``.
    """
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    cos_x = np.cos(x)
    base = d - np.sum(cos_x, axis=-1)
    term = base[..., None] + idx * (1.0 - cos_x) - np.sin(x)
    return np.sum(term**2, axis=-1)


def extended_feudenstein_and_roth(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Thesis formula (per pair):

    (-13 + x_p + ((5 - x_d) x_d - 2) x_d)^2 + (-29 + x_p + ((x_d + 1) x_d - 14) x_d)^2

    The frozen prototype uses ``((5 - x_d) x_d - 2) x_d`` in both terms; that variant is
    available as ``extended_feudenstein_and_roth_original``.
    """
    xp, xd = _pair_split(x)
    a = (5.0 - xd) * xd - 2.0
    b = (xd + 1.0) * xd - 14.0
    return np.sum((-13.0 + xp + a * xd) ** 2 + (-29.0 + xp + b * xd) ** 2, axis=-1)


def extended_baele(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per pair: (1.5 - x_p(1-x_d))^2 + (2.25 - x_p(1-x_d^2))^2 + (2.625 - x_p(1-x_d^3))^2."""
    xp, xd = _pair_split(x)
    t1 = (1.5 - xp * (1.0 - xd)) ** 2
    t2 = (2.25 - xp * (1.0 - xd**2)) ** 2
    t3 = (2.625 - xp * (1.0 - xd**3)) ** 2
    return np.sum(t1 + t2 + t3, axis=-1)


def extended_penalty(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_{i<n} (x_i - 1)^2 + (sum_i x_i^2 - 0.25)^2."""
    return np.sum((x[..., :-1] - 1.0) ** 2, axis=-1) + (np.sum(x**2, axis=-1) - 0.25) ** 2


def raydan_1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = (1/10) sum_i i (e^{x_i} - x_i)."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    return 0.1 * np.sum(idx * (np.exp(x) - x), axis=-1)


def raydan_2(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i (e^{x_i} - x_i)."""
    return np.sum(np.exp(x) - x, axis=-1)


def diagonal_1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i (e^{x_i} - i x_i)."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    return np.sum(np.exp(x) - idx * x, axis=-1)


def diagonal_2(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i (e^{x_i} - x_i / i)."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    return np.sum(np.exp(x) - x / idx, axis=-1)


def diagonal_3(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i (e^{x_i} - i sin x_i)."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    return np.sum(np.exp(x) - idx * np.sin(x), axis=-1)


def hager(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i (e^{x_i} - sqrt(i) x_i)."""
    d = x.shape[-1]
    idx = np.sqrt(np.arange(1, d + 1))
    return np.sum(np.exp(x) - idx * x, axis=-1)


def extended_tridiagonal_1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Thesis formula:

    F(x) = sum_{i=1}^{n-1} [(x_i + x_{i+1} - 3)^2 + (x_i - x_{i+1} + 1)^4]

    Note: the frozen prototype evaluates this only on disjoint pairs ``(x_{2i-1}, x_{2i})``;
    that variant is available as ``extended_tridiagonal_1_original``.
    """
    xp, xd = x[..., :-1], x[..., 1:]
    return np.sum((xp + xd - 3.0) ** 2 + (xp - xd + 1.0) ** 4, axis=-1)


def diagonal_4(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per pair: 0.5 (x_p^2 + c x_d^2), c = 100."""
    xp, xd = _pair_split(x)
    return 0.5 * np.sum(xp**2 + 100.0 * xd**2, axis=-1)


def diagonal_5(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i log(e^{x_i} + e^{-x_i})."""
    return np.sum(np.log(np.exp(x) + np.exp(-x)), axis=-1)


def extended_himmelblau(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per pair: (x_p^2 + x_d - 11)^2 + (x_p + x_d^2 - 7)^2."""
    xp, xd = _pair_split(x)
    return np.sum((xp**2 + xd - 11.0) ** 2 + (xp + xd**2 - 7.0) ** 2, axis=-1)


def extended_psc1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per pair: (x_p^2 + x_d^2 + x_p x_d)^2 + sin^2(x_p) + cos^2(x_d)."""
    xp, xd = _pair_split(x)
    t1 = (xp**2 + xd**2 + xp * xd) ** 2
    return np.sum(t1 + np.sin(xp) ** 2 + np.cos(xd) ** 2, axis=-1)


def sincos(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per pair: (x_p^2 + x_d^2 + x_p x_d)^2 + sin^2(x_p) + cos^2(x_d)."""
    xp, xd = _pair_split(x)
    t1 = (xp**2 + xd**2 + xp * xd) ** 2
    return np.sum(t1 + np.sin(xp) ** 2 + np.cos(xd) ** 2, axis=-1)


def extended_bd1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per pair: (x_p^2 + x_d - 2)^2 + (e^{x_p - 1} - x_p)^2."""
    xp, xd = _pair_split(x)
    return np.sum((xp**2 + xd - 2.0) ** 2 + (np.exp(xp - 1.0) - xp) ** 2, axis=-1)


def extended_maratos(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per pair: x_p + c (x_p^2 + x_d^2 - 1)^2, c = 100."""
    xp, xd = _pair_split(x)
    return np.sum(xp + 100.0 * (xp**2 + xd**2 - 1.0) ** 2, axis=-1)


def extended_cliff(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per pair: ((x_p - 3)/100)^2 + (x_p - x_d) + exp(20 (x_p - x_d))."""
    xp, xd = _pair_split(x)
    return np.sum(((xp - 3.0) / 100.0) ** 2 + (xp - xd) + np.exp(20.0 * (xp - xd)), axis=-1)


def extended_hiebert(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per pair: (x_p - 10)^2 + (x_p x_d - 50000)^2."""
    xp, xd = _pair_split(x)
    return np.sum((xp - 10.0) ** 2 + (xp * xd - 50000.0) ** 2, axis=-1)


def quadratic_qf1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = 0.5 sum_i i x_i^2 + x_n."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    return 0.5 * np.sum(idx * x**2, axis=-1) + x[..., -1]


def extended_quadratic_penalty_qp1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_{i<n} (x_i^2 - 2)^2 + (sum_i x_i^2 - 0.5)^2."""
    return np.sum((x[..., :-1] ** 2 - 2.0) ** 2, axis=-1) + (np.sum(x**2, axis=-1) - 0.5) ** 2


def extended_quadratic_penalty_qp2(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_{i<n} (x_i^2 - sin x_i)^2 + (sum_i x_i^2 - 100)^2."""
    prev = x[..., :-1]
    return np.sum((prev**2 - np.sin(prev)) ** 2, axis=-1) + (np.sum(x**2, axis=-1) - 100.0) ** 2


def quadratic_qf2(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = 0.5 sum_i i (x_i^2 - 1)^2 + x_n."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    return 0.5 * np.sum(idx * (x**2 - 1.0) ** 2, axis=-1) + x[..., -1]


def extended_quadratic_exponential_ep1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per pair: (e^{x_p - x_d} - 5)^2 + (x_p - x_d)^2 (x_p - x_d - 11)^2."""
    xp, xd = _pair_split(x)
    diff = xp - xd
    return np.sum((np.exp(diff) - 5.0) ** 2 + diff**2 * (diff - 11.0) ** 2, axis=-1)


def extended_tridiagonal_2(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Thesis formula:

    F(x) = sum_{i=1}^{n-1} [(x_i x_{i+1} - 1)^2 + 0.1 (x_i + 1)(x_{i+1} + 1)]

    Note: the frozen prototype uses ``0.1 (x_i + 1)^2`` in the second term; that variant is
    available as ``extended_tridiagonal_2_original``.
    """
    xp, xd = x[..., :-1], x[..., 1:]
    return np.sum((xp * xd - 1.0) ** 2 + 0.1 * (xp + 1.0) * (xd + 1.0), axis=-1)


def fletcbv3(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Fletcher C-V3: p = 1e-8, h = 1/(n+1), c = 1.

    0.5 p (x_1^2 + x_n^2) + sum_{i<n} (p/2)(x_i - x_{i+1})^2
    + sum_i [p (h^2 + 2)/h^2 x_i + c p/h^2 cos x_i].
    """
    d = x.shape[-1]
    p = 1e-8
    h = 1.0 / (d + 1)
    t1 = 0.5 * p * (x[..., 0] ** 2 + x[..., -1] ** 2)
    t2 = np.sum((p / 2.0) * (x[..., :-1] - x[..., 1:]) ** 2, axis=-1)
    t3 = np.sum(p * (h**2 + 2.0) / h**2 * x + p / h**2 * np.cos(x), axis=-1)
    return t1 + t2 + t3


def fletchcr(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_{i=1}^{n-1} c (x_{i+1} - x_i + 1 - x_i^2)^2, c = 100."""
    xp, xd = x[..., :-1], x[..., 1:]
    return np.sum(100.0 * (xd - xp + 1.0 - xp**2) ** 2, axis=-1)


def bdqrtic(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Broyden tridiagonal quartic (as in the frozen prototype).

    sum_{i=1}^{n-3} (-4 x_i + 3)^2
    + sum_{i=1}^{n-3} (x_i^2 + 2 x_{i+1}^2 + 3 x_{i+2}^2 + 4 x_{i+3}^2 + 5 x_n^2)^2
    """
    xs = x**2
    t1 = np.sum((-4.0 * x[..., :-3] + 3.0) ** 2, axis=-1)
    t2 = np.sum(
        (xs[..., :-3] + 2.0 * xs[..., 1:-2] + 3.0 * xs[..., 2:-1]
         + 4.0 * xs[..., 3:] + 5.0 * xs[..., -1][..., None]) ** 2,
        axis=-1,
    )
    return t1 + t2


def tridia(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""alpha=2, beta=gamma=delta=1: (x_1 - 1)^2 + sum_{i=2}^{n} i (2 x_i - x_{i-1})^2."""
    d = x.shape[-1]
    idx = np.arange(2, d + 1)
    t1 = (x[..., 0] - 1.0) ** 2
    t2 = np.sum(idx * (2.0 * x[..., 1:] - x[..., :-1]) ** 2, axis=-1)
    return t1 + t2


def arwhead(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""As in the frozen prototype: sum_{i<n} (-4 x_i + 3) + sum_{i<n} (x_i^2 + x_n^2)^2."""
    xs = x**2
    t1 = np.sum(-4.0 * x[..., :-1] + 3.0, axis=-1)
    t2 = np.sum((xs[..., :-1] + xs[..., -1][..., None]) ** 2, axis=-1)
    return t1 + t2


def nondia(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""As in the frozen prototype:

    F(x) = (x_1 - 1)^2 + 100 (sum_{i>=2} (x_1 - x_i^2))^2

    Note that the square applies to the whole sum, not to each term.
    """
    t1 = (x[..., 0] - 1.0) ** 2
    t2 = 100.0 * (np.sum(x[..., 0][..., None] - x[..., 1:] ** 2, axis=-1)) ** 2
    return t1 + t2


def nondquar(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = (x_1 - x_2)^2 + sum_{i<=n-2} (x_i + x_{i+1} + x_n)^4 + (x_{n-1} + x_n)^2."""
    t1 = (x[..., 0] - x[..., 1]) ** 2
    t2 = np.sum((x[..., :-2] + x[..., 1:-1] + x[..., -1][..., None]) ** 4, axis=-1)
    t3 = (x[..., -2] + x[..., -1]) ** 2
    return t1 + t2 + t3


def dqdrtic(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""With X_i = x_i^2: F(x) = sum_{i<=n-2} (X_i + 100 X_{i+1} + 100 X_{i+2})."""
    xs = x**2
    return np.sum(xs[..., :-2] + 100.0 * xs[..., 1:-1] + 100.0 * xs[..., 2:], axis=-1)


def eg2(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""As in the frozen prototype: sum_{i<n} sin(x_1 + x_i^2 - 1) + 0.5 sin(x_n^2)."""
    x0 = x[..., 0]
    xs = x**2
    t1 = np.sum(np.sin(x0[..., None] + xs[..., :-1] - 1.0), axis=-1)
    t2 = 0.5 * np.sin(xs[..., -1])
    return t1 + t2


def broyden_tridiagonal(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Broyden tridiagonal (as in the frozen prototype).

    (3 x_1 - 2 x_1^2)^2
    + sum_{i=2}^{n-1} (3 x_i - 2 x_i^2 - x_{i-1} - 2 x_{i+1} + 1)^2
    + (3 x_n - 2 x_n^2 - x_{n-1} + 1)^2
    """
    xs = x**2
    t1 = (3.0 * x[..., 0] - 2.0 * xs[..., 0]) ** 2
    t2 = np.sum(
        (3.0 * x[..., 1:-1] - 2.0 * xs[..., 1:-1] - x[..., :-2] - 2.0 * x[..., 2:] + 1.0) ** 2,
        axis=-1,
    )
    t3 = (3.0 * x[..., -1] - 2.0 * xs[..., -1] - x[..., -2] + 1.0) ** 2
    return t1 + t2 + t3


def liarwhd(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""As effectively defined by the frozen prototype (its second, overriding definition):

    F(x) = 4 sum_i (x_i^2 - x_1)^2 + sum_i (x_i - 1)^2
    """
    t1 = 4.0 * np.sum((x**2 - x[..., 0][..., None]) ** 2, axis=-1)
    t2 = np.sum((x - 1.0) ** 2, axis=-1)
    return t1 + t2


def engval1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""As in the frozen prototype: sum_{i<n} (x_i^2 + x_{i+1}^2)^2 + sum_{i<n} (-4 x_i + 3)."""
    xs = x**2
    t1 = np.sum((xs[..., :-1] + xs[..., 1:]) ** 2, axis=-1)
    t2 = np.sum(-4.0 * x[..., :-1] + 3.0, axis=-1)
    return t1 + t2


def edensch(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""As in the frozen prototype:

    F(x) = 16 + sum_{i<n} [(x_i-2)^4 + (x_i x_{i+1} + 2 x_{i+1})^2 + (x_{i+1}+1)^2]
    """
    xp, xd = x[..., :-1], x[..., 1:]
    t1 = (xp - 2.0) ** 4
    t2 = (xp * xd + 2.0 * xd) ** 2
    t3 = (xd + 1.0) ** 2
    return 16.0 + np.sum(t1 + t2 + t3, axis=-1)


def indef(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""As in the frozen prototype: sum_i x_i + 0.5 sum_{1<i<n} cos(2 x_i - x_n - x_1).

    (The inner cosine uses coordinates strictly between the first and the last.)
    """
    t1 = np.sum(x, axis=-1)
    t2 = 0.5 * np.sum(
        np.cos(2.0 * x[..., 1:-1] - x[..., -1][..., None] - x[..., 0][..., None]), axis=-1
    )
    return t1 + t2


def cube(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = (x_1 - 1)^2 + 100 sum_{i>=2} (x_i - x_{i-1}^3)^2."""
    xp, xd = x[..., :-1], x[..., 1:]
    return (x[..., 0] - 1.0) ** 2 + 100.0 * np.sum((xd - xp**3) ** 2, axis=-1)


def bdexp(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""As in the frozen prototype: sum_{i<=n-2} (x_i + x_{i+1}) exp(-(x_i + x_{i+1}) x_{i+2})."""
    t1 = x[..., :-2] + x[..., 1:-1]
    return np.sum(t1 * np.exp(-t1 * x[..., 2:]), axis=-1)


def genhumps(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""As in the frozen prototype:

    sum_{i<n} [sin^2(2 x_i) sin^2(2 x_{i+1}) + 0.05 (x_i^2 + x_{i+1}^2)]
    """
    xp, xd = x[..., :-1], x[..., 1:]
    t1 = np.sin(2.0 * xp) ** 2 * np.sin(2.0 * xd) ** 2
    t2 = 0.05 * (xp**2 + xd**2)
    return np.sum(t1 + t2, axis=-1)


def mccormck(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""As in the frozen prototype:

    sum_{i<n} [-1.5 x_i + 2.5 x_{i+1} + 1 + (x_i - x_{i+1})^2 + sin(x_i + x_{i+1})]
    """
    xp, xd = x[..., :-1], x[..., 1:]
    t1 = -1.5 * xp + 2.5 * xd + 1.0
    t2 = (xp - xd) ** 2
    t3 = np.sin(xp + xd)
    return np.sum(t1 + t2 + t3, axis=-1)


def nonscomp(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = (x_1 - 1)^2 + 4 sum_{i>=2} (x_i - x_{i-1}^2)^2."""
    xp, xd = x[..., :-1], x[..., 1:]
    return (x[..., 0] - 1.0) ** 2 + 4.0 * np.sum((xd - xp**2) ** 2, axis=-1)


def vardim(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""s = sum_i i x_i - n(n+1)/2; F(x) = sum_i (x_i - 1)^2 + s^2 + s^4."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    s = np.sum(idx * x, axis=-1) - d * (d + 1) / 2.0
    return np.sum((x - 1.0) ** 2, axis=-1) + s**2 + s**4


def quartc(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i (x_i - 1)^4."""
    return np.sum((x - 1.0) ** 4, axis=-1)


def sinquad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""As in the frozen prototype:

    (x_1 - 1)^4 + sum_{1<i<n} (sin(x_i - x_n) - x_1^2 + x_i^2)^2 + (x_n^2 - x_1^2)^2
    """
    xs = x**2
    t1 = (x[..., 0] - 1.0) ** 4
    t2 = np.sum(
        (np.sin(x[..., 1:-1] - x[..., -1][..., None]) - xs[..., 0][..., None] + xs[..., 1:-1]) ** 2,
        axis=-1,
    )
    t3 = (xs[..., -1] - xs[..., 0]) ** 2
    return t1 + t2 + t3


def extended_denschnb(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per pair: (x_p - 2)^2 + (x_p - 2)^2 x_d^2 + (x_d + 1)^2."""
    xp, xd = _pair_split(x)
    t1 = (xp - 2.0) ** 2
    return np.sum(t1 + t1 * xd**2 + (xd + 1.0) ** 2, axis=-1)


def extended_denschnf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per pair: [2(x_p+x_d)^2 + (x_p-x_d)^2 - 8]^2 + [5 x_p^2 + (x_p-3)^2 - 9]^2."""
    xp, xd = _pair_split(x)
    t1 = (2.0 * (xp + xd) ** 2 + (xp - xd) ** 2 - 8.0) ** 2
    t2 = (5.0 * xp**2 + (xp - 3.0) ** 2 - 9.0) ** 2
    return np.sum(t1 + t2, axis=-1)


def dixon3dq(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = (x_1 - 1)^2 + sum_{i<n} (x_i - x_{i+1})^2 + (x_n - 1)^2."""
    t1 = (x[..., 0] - 1.0) ** 2
    t2 = np.sum((x[..., :-1] - x[..., 1:]) ** 2, axis=-1)
    t3 = (x[..., -1] - 1.0) ** 2
    return t1 + t2 + t3


def cosine(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_{i<n} cos(-0.5 x_{i+1} + x_i^2)."""
    xp, xd = x[..., :-1], x[..., 1:]
    return np.sum(np.cos(-0.5 * xd + xp**2), axis=-1)


def sine(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_{i<n} sin(-0.5 x_{i+1} + x_i^2)."""
    xp, xd = x[..., :-1], x[..., 1:]
    return np.sum(np.sin(-0.5 * xd + xp**2), axis=-1)


def biggsb1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = (x_1 - 1)^2 + sum_{i<n} (x_i - x_{i+1})^2 + (x_n - 1)^2."""
    t1 = (x[..., 0] - 1.0) ** 2
    t2 = np.sum((x[..., :-1] - x[..., 1:]) ** 2, axis=-1)
    t3 = (x[..., -1] - 1.0) ** 2
    return t1 + t2 + t3


def generalized_quartic(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_{i=1}^{n-1} [x_i^2 + (x_{i+1} + x_i^2)^2]."""
    xp, xd = x[..., :-1], x[..., 1:]
    return np.sum(xp**2 + (xd + xp**2) ** 2, axis=-1)


def diagonal_7(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i (e^{x_i} - 2 x_i - x_i^2)."""
    return np.sum(np.exp(x) - 2.0 * x - x**2, axis=-1)


def diagonal_8(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i (x_i e^{x_i} - 2 x_i - x_i^2)."""
    return np.sum(x * np.exp(x) - 2.0 * x - x**2, axis=-1)


def fh3(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = (sum_i x_i)^2 + F_diagonal_8(x)."""
    return np.sum(x, axis=-1) ** 2 + diagonal_8(x)


def diagonal_9(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i (e^{x_i} - i x_i) + 10000 x_n^2."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    return np.sum(np.exp(x) - idx * x, axis=-1) + 10000.0 * x[..., -1] ** 2


def himmelbg(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per pair: (2 x_p^2 + 3 x_d^2) exp(-(x_p + x_d))."""
    xp, xd = _pair_split(x)
    return np.sum((2.0 * xp**2 + 3.0 * xd**2) * np.exp(-xp - xd), axis=-1)


def himmelh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Per pair: x_p^3 + x_d^2 - 3 x_p - 2 x_d + 2."""
    xp, xd = _pair_split(x)
    return np.sum(xp**3 + xd**2 - 3.0 * xp - 2.0 * xd + 2.0, axis=-1)


def ackley(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Ackley with a=20, b=0.2, c=2 pi."""
    t1 = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2, axis=-1)))
    t2 = -np.exp(np.mean(np.cos(2.0 * np.pi * x), axis=-1))
    return t1 + t2 + 20.0 + np.e


def griewank(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Griewank: sum_i x_i^2/4000 - prod_i cos(x_i / sqrt(i)) + 1."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    t1 = np.sum(x**2, axis=-1) / 4000.0
    t2 = np.prod(np.cos(x / np.sqrt(idx)), axis=-1)
    return t1 - t2 + 1.0


def levy(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Levy with w_i = 1 + (x_i - 1)/4 (minimum 0 at x = ones)."""
    w = 1.0 + (x - 1.0) / 4.0
    t1 = np.sin(np.pi * w[..., 0]) ** 2
    mid = w[..., :-1]
    t2 = np.sum((mid - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * mid + 1.0) ** 2), axis=-1)
    wl = w[..., -1]
    t3 = (wl - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * wl) ** 2)
    return t1 + t2 + t3


def rastrigin(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Rastrigin: 10 n + sum_i (x_i^2 - 10 cos(2 pi x_i))."""
    d = x.shape[-1]
    return 10.0 * d + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=-1)


def schwefel(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Schwefel: 418.9829 n - sum_i x_i sin(sqrt(|x_i|))."""
    d = x.shape[-1]
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)


def sum_of_different_powers(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""F(x) = sum_i |x_i|^{i+1}."""
    d = x.shape[-1]
    idx = np.arange(2, d + 2)
    return np.sum(np.abs(x)**idx, axis=-1)


def trid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Thesis/classical More-Sugrue Trid:

    F(x) = sum_i (x_i - 1)^2 - sum_{i=2}^{n} x_i x_{i-1}

    Note: the frozen prototype adds the second term instead of subtracting it; that variant is
    available as ``trid_original``.
    """
    xp, xd = x[..., :-1], x[..., 1:]
    return np.sum((x - 1.0) ** 2, axis=-1) - np.sum(xp * xd, axis=-1)


def zakharov(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Zakharov: s = sum_i 0.5 i x_i; F(x) = sum_i x_i^2 + s^2 + s^4."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    s = np.sum(0.5 * idx * x, axis=-1)
    return np.sum(x**2, axis=-1) + s**2 + s**4


# ---------------------------------------------------------------------------
# Legacy variants (frozen-prototype behavior, kept for parity studies only)
# ---------------------------------------------------------------------------


def extended_trigonometric_original(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Frozen-prototype variant: same as ``extended_trigonometric`` but with ``+ sin x_i``."""
    d = x.shape[-1]
    idx = np.arange(1, d + 1)
    cos_x = np.cos(x)
    base = d - np.sum(cos_x, axis=-1)
    term = base[..., None] + idx * (1.0 - cos_x) + np.sin(x)
    return np.sum(term**2, axis=-1)


def extended_feudenstein_and_roth_original(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Frozen-prototype variant: both terms use ``(5 - x_d) x_d - 2``."""
    xp, xd = _pair_split(x)
    a = (5.0 - xd) * xd - 2.0
    return np.sum((-13.0 + xp + a * xd) ** 2 + (-29.0 + xp + a * xd) ** 2, axis=-1)


def extended_tridiagonal_1_original(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Frozen-prototype variant: pair-based (disjoint pairs) instead of consecutive chain."""
    xp, xd = _pair_split(x)
    return np.sum((xp + xd - 3.0) ** 2 + (xp - xd + 1.0) ** 4, axis=-1)


def extended_tridiagonal_2_original(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Frozen-prototype variant: second term is ``0.1 (x_i + 1)^2`` (only the first factor)."""
    xp, xd = x[..., :-1], x[..., 1:]
    return np.sum((xp * xd - 1.0) ** 2 + 0.1 * (xp + 1.0) ** 2, axis=-1)


def trid_original(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Frozen-prototype variant: ``sum_i (x_i - 1)^2 + sum_{i=2}^{n} x_i x_{i-1}``."""
    xp, xd = x[..., :-1], x[..., 1:]
    return np.sum((x - 1.0) ** 2, axis=-1) + np.sum(xp * xd, axis=-1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: Canonical benchmark names in the order of ``original_python_code/functions.txt``.
FUNCTION_NAMES: tuple[str, ...] = (
    "extended_feudenstein_and_roth",
    "extended_trigonometric",
    "extended_rosenbrock",
    "generalized_rosenbrock",
    "extended_white_and_holst",
    "extended_baele",
    "extended_penalty",
    "perturbed_quadratic",
    "raydan_1",
    "raydan_2",
    "diagonal_1",
    "diagonal_2",
    "diagonal_3",
    "hager",
    "extended_tridiagonal_1",
    "diagonal_4",
    "diagonal_5",
    "extended_himmelblau",
    "generalized_white_and_holst",
    "extended_psc1",
    "extended_bd1",
    "extended_maratos",
    "extended_cliff",
    "perturbed_quadratic_diagonal",
    "extended_hiebert",
    "quadratic_qf1",
    "extended_quadratic_penalty_qp1",
    "extended_quadratic_penalty_qp2",
    "quadratic_qf2",
    "extended_quadratic_exponential_ep1",
    "extended_tridiagonal_2",
    "fletcbv3",
    "fletchcr",
    "bdqrtic",
    "tridia",
    "arwhead",
    "nondia",
    "nondquar",
    "dqdrtic",
    "eg2",
    "broyden_tridiagonal",
    "almost_perturbed_quadratic",
    "liarwhd",
    "power",
    "engval1",
    "edensch",
    "indef",
    "cube",
    "bdexp",
    "genhumps",
    "mccormck",
    "nonscomp",
    "vardim",
    "quartc",
    "sinquad",
    "extended_denschnb",
    "extended_denschnf",
    "dixon3dq",
    "cosine",
    "sine",
    "biggsb1",
    "generalized_quartic",
    "diagonal_7",
    "diagonal_8",
    "fh3",
    "sincos",
    "diagonal_9",
    "himmelbg",
    "himmelh",
    "ackley",
    "griewank",
    "levy",
    "rastrigin",
    "schwefel",
    "sphere",
    "sum_of_different_powers",
    "trid",
    "zakharov",
)

#: Legacy variants reproducing frozen-prototype behavior (parity studies only).
LEGACY_VARIANTS: dict[str, VectorizedFn] = {
    "extended_trigonometric_original": extended_trigonometric_original,
    "extended_feudenstein_and_roth_original": extended_feudenstein_and_roth_original,
    "extended_tridiagonal_1_original": extended_tridiagonal_1_original,
    "extended_tridiagonal_2_original": extended_tridiagonal_2_original,
    "trid_original": trid_original,
}

#: Functions starred in ``original_python_code/functions.txt`` (used by the thesis experiments).
THESIS_STARRED_FUNCTIONS: tuple[str, ...] = (
    "extended_feudenstein_and_roth",
    "extended_trigonometric",
    "extended_rosenbrock",
    "generalized_rosenbrock",
    "extended_white_and_holst",
    "perturbed_quadratic",
    "extended_tridiagonal_1",
    "extended_himmelblau",
    "generalized_white_and_holst",
    "extended_psc1",
    "perturbed_quadratic_diagonal",
    "extended_hiebert",
    "extended_tridiagonal_2",
    "almost_perturbed_quadratic",
    "power",
    "cube",
    "generalized_quartic",
    "ackley",
    "griewank",
    "levy",
    "rastrigin",
    "schwefel",
    "sphere",
    "sum_of_different_powers",
    "trid",
    "zakharov",
)

_REGISTRY: dict[str, VectorizedFn] = {name: globals()[name] for name in FUNCTION_NAMES}
_REGISTRY.update(LEGACY_VARIANTS)


class Function:
    """A named batch-evaluable objective function.

    Parameters
    ----------
    name:
        Registry key; either a canonical name or a ``*_original`` legacy variant.
    """

    def __init__(self, name: str) -> None:
        if name not in _REGISTRY:
            known = ", ".join(sorted(_REGISTRY))
            raise KeyError(f"Unknown function '{name}'. Known functions: {known}")
        self.name = name
        self._fn = _REGISTRY[name]

    @property
    def evaluate_batched(self) -> VectorizedFn:
        """The underlying vectorized callable ``( ..., dim ) -> ( ..., )``."""
        return self._fn

    def evaluate(self, points: np.ndarray) -> float | NDArray[np.float64]:
        """Evaluate at one point of shape ``(dim,)``, returning a scalar, or at a batch of
        shape ``(n, dim)``, returning an array of shape ``(n,)``."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            return float(self._fn(pts[None, :])[0])
        return self._fn(pts)


def get_function(name: str) -> Function:
    """Look up a benchmark function by registry name."""
    return Function(name)


class CountingFunction:
    """Wraps a batch-evaluable callable and counts scalar function evaluations.

    A call with a single point of shape ``(dim,)`` counts as 1 evaluation; a call with a
    batch of shape ``(n, dim)`` counts as ``n`` evaluations.
    """

    def __init__(self, fn: VectorizedFn) -> None:
        self._fn = fn
        self.n_evaluations = 0

    def __call__(self, points: np.ndarray) -> float | NDArray[np.float64]:
        pts = np.asarray(points, dtype=np.float64)
        n = 1 if pts.ndim == 1 else int(np.prod(pts.shape[:-1]))
        out = self._fn(pts if pts.ndim > 1 else pts[None, :])
        self.n_evaluations += n
        return out[0] if pts.ndim == 1 else out
