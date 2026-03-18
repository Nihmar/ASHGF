"""
Optimized benchmark functions — tuned for N in [10, 1000].

Strategy (empirically verified, see crossover analysis):
  - Numba @njit(cache=True) wins over NumPy and numexpr for ALL functions
    across the full 10–1000 range.
  - Crossover to NumPy only occurs around N ≈ 5000 for transcendental-heavy
    functions; numexpr never wins below N=10000 due to call overhead.
  - cache=True writes compiled bytecode to __pycache__ so the JIT cost is
    paid once across the lifetime of the project, not per run.

Key micro-optimizations applied:
  1. Zero temporary allocations  — scalar loops, no intermediate arrays
  2. Subexpression reuse         — compute (x_p - x_d) once, not twice
  3. Fused loops                 — multiple reductions in a single pass
  4. np.dot(x, x) for ‖x‖²      — single BLAS call, no squared copy
  5. Avoid in-place x = x**2     — prevents accidental aliasing bugs
"""

import math
from functools import lru_cache

import numpy as np
from numba import njit

# Crossover thresholds measured empirically (see crossover_full.py):
#   exp-simple   (raydan_2, diagonal_5/7/8, fh3):       numpy wins at N > 1500
#   exp-indexed  (diagonal_1/2/3/9, raydan_1, hager):   numpy wins at N > 1000
#   cos-prod     (griewank):                            numpy wins at N > 1000
#   sin-cos      (rastrigin, ackley, schwefel, levy…):  numba wins up to N=10000
_THRESH_EXP_SIMPLE = 1500
_THRESH_EXP_INDEXED = 1000
_THRESH_COS_PROD = 1000


@lru_cache(maxsize=1024)
def _arange(n: int) -> np.ndarray:
    return np.arange(1, n + 1, dtype=np.float64)


# ---------------------------------------------------------------------------
# Pure arithmetic
# ---------------------------------------------------------------------------


@njit(cache=True)
def sphere(x):
    return np.dot(x, x)


@njit(cache=True)
def power(x):
    s = 0.0
    for i in range(len(x)):
        s += (i + 1) * x[i] * x[i]
    return s


@njit(cache=True)
def extended_rosenbrock(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        a = x[i]
        b = x[i + 1]
        d = b - a * a
        s += 100.0 * d * d + (1.0 - a) ** 2
    return s


@njit(cache=True)
def generalized_rosenbrock(x):
    s = 0.0
    for i in range(len(x) - 1):
        d = x[i + 1] - x[i] * x[i]
        s += 100.0 * d * d + (1.0 - x[i]) ** 2
    return s


@njit(cache=True)
def extended_white_and_holst(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        xp = x[i]
        xd = x[i + 1]
        d = xd - xp * xp * xp
        s += 100.0 * d * d + (1.0 - xp) ** 2
    return s


@njit(cache=True)
def generalized_white_and_holst(x):
    s = 0.0
    for i in range(len(x) - 1):
        d = x[i + 1] - x[i] ** 3
        s += 100.0 * d * d + (1.0 - x[i]) ** 2
    return s


@njit(cache=True)
def extended_feudenstein_and_roth(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        xp = x[i]
        xd = x[i + 1]
        poly = ((5.0 - xd) * xd - 2.0) * xd
        s += (-13.0 + xp + poly) ** 2 + (-29.0 + xp + poly) ** 2
    return s


@njit(cache=True)
def extended_baele(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        xp = x[i]
        xd = x[i + 1]
        s += (1.5 - xp * (1.0 - xd)) ** 2
        s += (2.25 - xp * (1.0 - xd * xd)) ** 2
        s += (2.625 - xp * (1.0 - xd**3)) ** 2
    return s


@njit(cache=True)
def extended_penalty(x):
    s1 = 0.0
    sq = 0.0
    for i in range(len(x) - 1):
        d = x[i] - 1.0
        s1 += d * d
        sq += x[i] * x[i]
    sq += x[-1] * x[-1]
    return s1 + (sq - 0.25) ** 2


@njit(cache=True)
def perturbed_quadratic(x):
    s = 0.0
    total = 0.0
    for i in range(len(x)):
        xi = x[i]
        s += (i + 1) * xi * xi
        total += xi
    return s + 0.01 * total * total


@njit(cache=True)
def almost_perturbed_quadratic(x):
    s = 0.0
    for i in range(len(x)):
        s += (i + 1) * x[i] * x[i]
    return s + 0.01 * (x[0] + x[-1]) ** 2


@njit(cache=True)
def perturbed_quadratic_diagonal(x):
    total = 0.0
    sq_sum = 0.0
    for i in range(len(x)):
        total += x[i]
        sq_sum += (i + 1) * x[i] * x[i]
    return total * total + sq_sum / 100.0


@njit(cache=True)
def generalized_tridiagonal_1(x):
    s = 0.0
    for i in range(len(x) - 1):
        a = x[i] + x[i + 1] - 3.0
        b = x[i] - x[i + 1] + 1.0
        s += a * a + b * b * b * b
    return s


@njit(cache=True)
def extended_tridiagonal_1(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        a = x[i] + x[i + 1] - 3.0
        b = x[i] - x[i + 1] + 1.0
        s += a * a + b * b * b * b
    return s


@njit(cache=True)
def diagonal_4(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        s += x[i] * x[i] + 100.0 * x[i + 1] * x[i + 1]
    return 0.5 * s


@njit(cache=True)
def extended_himmelblau(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        a = x[i] * x[i] + x[i + 1] - 11.0
        b = x[i] + x[i + 1] * x[i + 1] - 7.0
        s += a * a + b * b
    return s


@njit(cache=True)
def extended_psc1(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        xp = x[i]
        xd = x[i + 1]
        quad = xp * xp + xd * xd + xp * xd
        s += quad * quad + math.sin(xp) ** 2 + math.cos(xd) ** 2
    return s


sincos = extended_psc1  # identical formula


@njit(cache=True)
def extended_bd1(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        xp = x[i]
        xd = x[i + 1]
        a = xp * xp + xd - 2.0
        b = math.exp(xp - 1.0) - xp
        s += a * a + b * b
    return s


@njit(cache=True)
def extended_maratos(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        xp = x[i]
        xd = x[i + 1]
        d = xp * xp + xd * xd - 1.0
        s += xp + 100.0 * d * d
    return s


@njit(cache=True)
def extended_cliff(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        xp = x[i]
        xd = x[i + 1]
        diff = xp - xd
        s += ((xp - 3.0) / 100.0) ** 2 + diff + math.exp(20.0 * diff)
    return s


@njit(cache=True)
def extended_hiebert(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        xp = x[i]
        xd = x[i + 1]
        s += (xp - 10.0) ** 2 + (xp * xd - 50000.0) ** 2
    return s


@njit(cache=True)
def quadratic_qf1(x):
    s = 0.0
    for i in range(len(x)):
        s += (i + 1) * x[i] * x[i]
    return 0.5 * s + x[-1]


@njit(cache=True)
def quadratic_qf2(x):
    s = 0.0
    for i in range(len(x)):
        d = x[i] * x[i] - 1.0
        s += (i + 1) * d * d
    return 0.5 * s + x[-1]


@njit(cache=True)
def extended_quadratic_penalty_qp1(x):
    s1 = 0.0
    sq = 0.0
    for i in range(len(x) - 1):
        d = x[i] * x[i] - 2.0
        s1 += d * d
        sq += x[i] * x[i]
    sq += x[-1] * x[-1]
    return s1 + (sq - 0.5) ** 2


@njit(cache=True)
def extended_quadratic_penalty_qp2(x):
    s1 = 0.0
    sq = 0.0
    for i in range(len(x) - 1):
        xi = x[i]
        d = xi * xi - math.sin(xi)
        s1 += d * d
        sq += xi * xi
    sq += x[-1] * x[-1]
    return s1 + (sq - 100.0) ** 2


@njit(cache=True)
def extended_quadratic_exponential_ep1(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        d = x[i] - x[i + 1]
        s += (math.exp(d) - 5.0) ** 2 + d * d * (d - 11.0) ** 2
    return s


@njit(cache=True)
def extended_tridiagonal_2(x):
    s = 0.0
    for i in range(len(x) - 1):
        a = x[i + 1] * x[i] - 1.0
        b = x[i] + 1.0
        s += a * a + 0.1 * b * b
    return s


@njit(cache=True)
def fletchcr(x):
    s = 0.0
    for i in range(len(x) - 1):
        d = x[i + 1] - x[i] + 1.0 - x[i] * x[i]
        s += 100.0 * d * d
    return s


@njit(cache=True)
def tridia(x):
    s = (x[0] - 1.0) ** 2
    for i in range(1, len(x)):
        d = 2.0 * x[i] - x[i - 1]
        s += (i + 1) * d * d
    return s


@njit(cache=True)
def arwhead(x):
    s1 = 0.0
    s2 = 0.0
    x_last_sq = x[-1] * x[-1]
    for i in range(len(x) - 1):
        s1 += -4.0 * x[i] + 3.0
        d = x[i] * x[i] + x_last_sq
        s2 += d * d
    return s1 + s2


@njit(cache=True)
def nondia(x):
    s = (x[0] - 1.0) ** 2
    total = 0.0
    for i in range(1, len(x)):
        total += x[0] - x[i] * x[i]
    return s + 100.0 * total * total


@njit(cache=True)
def nondquar(x):
    s = (x[0] - x[1]) ** 2
    for i in range(len(x) - 2):
        d = x[i] + x[i + 1] + x[-1]
        s += d * d * d * d
    s += (x[-2] + x[-1]) ** 2
    return s


@njit(cache=True)
def dqdrtic(x):
    s = 0.0
    for i in range(len(x) - 2):
        s += x[i] ** 2 + 100.0 * x[i + 1] ** 2 + 100.0 * x[i + 2] ** 2
    return s


@njit(cache=True)
def broyden_tridiagonal(x):
    n = len(x)
    a = x[0]
    a2 = a * a
    s = (3.0 * a - 2.0 * a2) ** 2
    for i in range(1, n - 1):
        xi = x[i]
        xi2 = xi * xi
        d = 3.0 * xi - 2.0 * xi2 - x[i - 1] - 2.0 * x[i + 1] + 1.0
        s += d * d
    b = x[-1]
    b2 = b * b
    s += (3.0 * b - 2.0 * b2 - x[-2] + 1.0) ** 2
    return s


@njit(cache=True)
def liarwhd(x):
    x0 = x[0]
    s = 0.0
    for xi in x:
        s += 4.0 * (xi * xi - x0) ** 2 + (xi - 1.0) ** 2
    return s


@njit(cache=True)
def engval1(x):
    s = 0.0
    for i in range(len(x) - 1):
        d = x[i] * x[i] + x[i + 1] * x[i + 1]
        s += d * d - 4.0 * x[i] + 3.0
    return s


@njit(cache=True)
def edensch(x):
    s = 16.0
    for i in range(len(x) - 1):
        xi = x[i]
        xi1 = x[i + 1]
        s += (xi - 2.0) ** 4 + (xi * xi1 + 2.0 * xi1) ** 2 + (xi1 + 1.0) ** 2
    return s


@njit(cache=True)
def cube(x):
    s = (x[0] - 1.0) ** 2
    for i in range(1, len(x)):
        d = x[i] - x[i - 1] ** 3
        s += 100.0 * d * d
    return s


@njit(cache=True)
def nonscomp(x):
    s = (x[0] - 1.0) ** 2
    for i in range(1, len(x)):
        d = x[i] - x[i - 1] ** 2
        s += 4.0 * d * d
    return s


@njit(cache=True)
def vardim(x):
    n = len(x)
    s1 = 0.0
    s2 = 0.0
    for i in range(n):
        s1 += (x[i] - 1.0) ** 2
        s2 += (i + 1) * x[i]
    t = s2 - n * (n + 1) / 2
    return s1 + t * t + t * t * t * t


@njit(cache=True)
def quartc(x):
    s = 0.0
    for xi in x:
        d = xi - 1.0
        s += d * d * d * d
    return s


@njit(cache=True)
def sinquad(x):
    n = len(x)
    x0sq = x[0] * x[0]
    xnsq = x[-1] * x[-1]
    s = (x[0] - 1.0) ** 4
    for i in range(1, n - 1):
        xisq = x[i] * x[i]
        d = math.sin(x[i] - x[-1]) - x0sq + xisq
        s += d * d
    s += (xnsq - x0sq) ** 2
    return s


@njit(cache=True)
def extended_denschnb(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        xp = x[i]
        xd = x[i + 1]
        d = (xp - 2.0) ** 2
        s += d + d * xd * xd + (xd + 1.0) ** 2
    return s


@njit(cache=True)
def extended_denschnf(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        xp = x[i]
        xd = x[i + 1]
        t1 = 2.0 * (xp + xd) ** 2 + (xp - xd) ** 2 - 8.0
        s += t1 * t1 + (5.0 * xp * xp + (xp - 3.0) ** 2 - 9.0) ** 2
    return s


@njit(cache=True)
def dixon3dq(x):
    s = (x[0] - 1.0) ** 2 + (x[-1] - 1.0) ** 2
    for i in range(len(x) - 1):
        d = x[i] - x[i + 1]
        s += d * d
    return s


@njit(cache=True)
def biggsb1(x):
    s = (x[0] - 1.0) ** 2 + (x[-1] - 1.0) ** 2
    for i in range(len(x) - 1):
        d = x[i + 1] - x[i]
        s += d * d
    return s


@njit(cache=True)
def generalized_quartic(x):
    s = 0.0
    for i in range(len(x) - 1):
        xi2 = x[i] * x[i]
        d = x[i + 1] + xi2
        s += xi2 + d * d
    return s


@njit(cache=True)
def himmelbg(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        xp = x[i]
        xd = x[i + 1]
        s += (2.0 * xp * xp + 3.0 * xd * xd) * math.exp(-xp - xd)
    return s


@njit(cache=True)
def himmelh(x):
    s = 0.0
    for i in range(0, len(x) - 1, 2):
        xp = x[i]
        xd = x[i + 1]
        s += -3.0 * xp - 2.0 * xd + 2.0 + xp**3 + xd * xd
    return s


@njit(cache=True)
def trid(x):
    s1 = 0.0
    s2 = 0.0
    for i in range(len(x)):
        s1 += (x[i] - 1.0) ** 2
    for i in range(1, len(x)):
        s2 += x[i] * x[i - 1]
    return s1 - s2


@njit(cache=True)
def zakharov(x):
    s1 = 0.0
    s2 = 0.0
    for i in range(len(x)):
        s1 += x[i] * x[i]
        s2 += 0.5 * (i + 1) * x[i]
    return s1 + s2 * s2 + s2 * s2 * s2 * s2


@njit(cache=True)
def sum_of_different_powers(x):
    s = 0.0
    for i in range(len(x)):
        s += abs(x[i]) ** (i + 2)
    return s


@njit(cache=True)
def cosine(x):
    s = 0.0
    for i in range(len(x) - 1):
        s += math.cos(-0.5 * x[i + 1] + x[i] * x[i])
    return s


@njit(cache=True)
def sine(x):
    s = 0.0
    for i in range(len(x) - 1):
        s += math.sin(-0.5 * x[i + 1] + x[i] * x[i])
    return s


# ---------------------------------------------------------------------------
# Transcendental — still @njit; crossover to NumPy only at N ≈ 5000
# ---------------------------------------------------------------------------

# --- exp-indexed group (crossover at N ≈ 1000) ---


@njit(cache=True)
def _raydan_1_jit(x):
    s = 0.0
    for i in range(len(x)):
        s += (i + 1) * (math.exp(x[i]) - x[i])
    return 0.1 * s


def raydan_1(x):
    if len(x) <= _THRESH_EXP_INDEXED:
        return _raydan_1_jit(x)
    i = _arange(len(x))
    return float(np.sum(i * (np.exp(x) - x)) * 0.1)


@njit(cache=True)
def _diagonal_1_jit(x):
    s = 0.0
    for i in range(len(x)):
        s += math.exp(x[i]) - (i + 1) * x[i]
    return s


def diagonal_1(x):
    if len(x) <= _THRESH_EXP_INDEXED:
        return _diagonal_1_jit(x)
    i = _arange(len(x))
    return float(np.sum(np.exp(x) - i * x))


@njit(cache=True)
def _diagonal_2_jit(x):
    s = 0.0
    for i in range(len(x)):
        s += math.exp(x[i]) - x[i] / (i + 1)
    return s


def diagonal_2(x):
    if len(x) <= _THRESH_EXP_INDEXED:
        return _diagonal_2_jit(x)
    i = _arange(len(x))
    return float(np.sum(np.exp(x) - x / i))


@njit(cache=True)
def _diagonal_3_jit(x):
    s = 0.0
    for i in range(len(x)):
        s += math.exp(x[i]) - (i + 1) * math.sin(x[i])
    return s


def diagonal_3(x):
    if len(x) <= _THRESH_EXP_INDEXED:
        return _diagonal_3_jit(x)
    i = _arange(len(x))
    return float(np.sum(np.exp(x) - i * np.sin(x)))


@njit(cache=True)
def _hager_jit(x):
    s = 0.0
    for i in range(len(x)):
        s += math.exp(x[i]) - math.sqrt(i + 1) * x[i]
    return s


def hager(x):
    if len(x) <= _THRESH_EXP_INDEXED:
        return _hager_jit(x)
    i = _arange(len(x))
    return float(np.sum(np.exp(x) - np.sqrt(i) * x))


@njit(cache=True)
def _diagonal_9_jit(x):
    s = 0.0
    for i in range(len(x)):
        s += math.exp(x[i]) - (i + 1) * x[i]
    return s + 10000.0 * x[-1] * x[-1]


def diagonal_9(x):
    if len(x) <= _THRESH_EXP_INDEXED:
        return _diagonal_9_jit(x)
    i = _arange(len(x))
    return float(np.sum(np.exp(x) - i * x)) + 10000.0 * x[-1] ** 2


# --- exp-simple group (crossover at N ≈ 1500) ---


@njit(cache=True)
def _raydan_2_jit(x):
    s = 0.0
    for xi in x:
        s += math.exp(xi) - xi
    return s


def raydan_2(x):
    if len(x) <= _THRESH_EXP_SIMPLE:
        return _raydan_2_jit(x)
    return float(np.sum(np.exp(x) - x))


@njit(cache=True)
def _diagonal_5_jit(x):
    s = 0.0
    for xi in x:
        s += math.log(math.exp(xi) + math.exp(-xi))
    return s


def diagonal_5(x):
    if len(x) <= _THRESH_EXP_SIMPLE:
        return _diagonal_5_jit(x)
    return float(np.sum(np.log(np.exp(x) + np.exp(-x))))


@njit(cache=True)
def _diagonal_7_jit(x):
    s = 0.0
    for xi in x:
        s += math.exp(xi) - 2.0 * xi - xi * xi
    return s


def diagonal_7(x):
    if len(x) <= _THRESH_EXP_SIMPLE:
        return _diagonal_7_jit(x)
    return float(np.sum(np.exp(x) - 2.0 * x - x**2))


@njit(cache=True)
def _diagonal_8_jit(x):
    s = 0.0
    for xi in x:
        s += xi * math.exp(xi) - 2.0 * xi - xi * xi
    return s


def diagonal_8(x):
    if len(x) <= _THRESH_EXP_SIMPLE:
        return _diagonal_8_jit(x)
    return float(np.sum(x * np.exp(x) - 2.0 * x - x**2))


@njit(cache=True)
def _fh3_jit(x):
    total = 0.0
    for xi in x:
        total += xi
    s = total * total
    for xi in x:
        s += xi * math.exp(xi) - 2.0 * xi - xi * xi
    return s


def fh3(x):
    if len(x) <= _THRESH_EXP_SIMPLE:
        return _fh3_jit(x)
    total = float(np.sum(x))
    return total * total + float(np.sum(x * np.exp(x) - 2.0 * x - x**2))


@njit(cache=True)
def extended_trigonometric(x):
    n = len(x)
    cos_sum = 0.0
    for xi in x:
        cos_sum += math.cos(xi)
    base = n - cos_sum
    s = 0.0
    for i in range(n):
        term = base + (i + 1) * (1.0 - math.cos(x[i])) + math.sin(x[i])
        s += term * term
    return s


@njit(cache=True)
def eg2(x):
    x0 = x[0]
    s = 0.0
    for i in range(len(x) - 1):
        s += math.sin(x0 + x[i] * x[i] - 1.0)
    s += 0.5 * math.sin(x[-1] * x[-1])
    return s


@njit(cache=True)
def indef(x):
    n = len(x)
    s = 0.0
    for xi in x:
        s += xi
    t = 0.0
    x0 = x[0]
    xn = x[-1]
    for i in range(1, n - 1):
        t += math.cos(2.0 * x[i] - xn - x0)
    return s + 0.5 * t


@njit(cache=True)
def genhumps(x):
    s = 0.0
    for i in range(len(x) - 1):
        s += math.sin(2.0 * x[i]) ** 2 * math.sin(2.0 * x[i + 1]) ** 2 + 0.05 * (
            x[i] ** 2 + x[i + 1] ** 2
        )
    return s


@njit(cache=True)
def mccormck(x):
    s = 0.0
    for i in range(len(x) - 1):
        a = x[i]
        b = x[i + 1]
        s += -1.5 * a + 2.5 * b + 1.0 + (a - b) ** 2 + math.sin(a + b)
    return s


@njit(cache=True)
def ackley(x):
    n = len(x)
    sq_sum = 0.0
    cos_sum = 0.0
    c = 2.0 * math.pi
    for xi in x:
        sq_sum += xi * xi
        cos_sum += math.cos(c * xi)
    t1 = -20.0 * math.exp(-0.2 * math.sqrt(sq_sum / n))
    t2 = -math.exp(cos_sum / n)
    return t1 + t2 + 20.0 + math.e


@njit(cache=True)
def _griewank_jit(x):
    sq = 0.0
    prod = 1.0
    for i in range(len(x)):
        sq += x[i] * x[i]
        prod *= math.cos(x[i] / math.sqrt(i + 1))
    return sq / 4000.0 - prod + 1.0


def griewank(x):
    if len(x) < _THRESH_COS_PROD:
        return _griewank_jit(x)
    i = _arange(len(x))
    return float(np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0)


@njit(cache=True)
def levy(x):
    n = len(x)
    pi = math.pi

    def w(xi):
        return 1.0 + (xi - 1.0) / 4.0

    w0 = w(x[0])
    s = math.sin(pi * w0) ** 2
    for i in range(n - 1):
        wi = w(x[i])
        s += (wi - 1.0) ** 2 * (1.0 + 10.0 * math.sin(pi * wi + 1.0) ** 2)
    wd = w(x[-1])
    s += (wd - 1.0) ** 2 * (1.0 + math.sin(2.0 * pi * wd) ** 2)
    return s


@njit(cache=True)
def rastrigin(x):
    c = 2.0 * math.pi
    s = 10.0 * len(x)
    for xi in x:
        s += xi * xi - 10.0 * math.cos(c * xi)
    return s


@njit(cache=True)
def schwefel(x):
    s = 0.0
    for xi in x:
        s += xi * math.sin(math.sqrt(abs(xi)))
    return 418.9829 * len(x) - s


# ---------------------------------------------------------------------------
# Special cases — structure prevents pure @njit; NumPy kept intentionally
# ---------------------------------------------------------------------------


def fletcbv3(x):
    """
    Three-term sum with boundary conditions. NumPy is fast enough and the
    formula maps naturally to array ops. No temporary blow-up here.
    """
    n = len(x)
    p = 1e-8
    h = 1.0 / (n + 1)
    factor = p * (h * h + 2.0) / (h * h)
    c_over_h2 = p / (h * h)
    diff = x[:-1] - x[1:]
    s = 0.5 * p * (x[0] ** 2 + x[-1] ** 2)
    s += 0.5 * p * np.dot(diff, diff)
    s += factor * np.sum(x) + c_over_h2 * np.sum(np.cos(x))
    return float(s)


def bdqrtic(x):
    """
    Stencil-based: accesses x[i], x[i+1], x[i+2], x[i+3] simultaneously.
    NumPy slice arithmetic is cleaner and fast for this pattern.
    """
    term_1 = np.sum((-4.0 * x[:-3] + 3.0) ** 2)
    x2 = x * x
    term_2 = np.sum(
        (x2[:-3] + 2.0 * x2[1:-2] + 3.0 * x2[2:-1] + 4.0 * x2[3:] + 5.0 * x2[-1]) ** 2
    )
    return float(term_1 + term_2)


@njit(cache=True)
def bdexp(x):
    s = 0.0
    for i in range(len(x) - 2):
        t = x[i] + x[i + 1]
        s += t * math.exp(-x[i + 2] * t)
    return s
