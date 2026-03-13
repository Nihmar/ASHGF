import math
import numpy as np


def extended_feudenstein_and_roth(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]

    term_1 = (-13 + x_p + ((5 - x_d) * x_d - 2) * x_d)**2
    term_2 = (-29 + x_p + ((5 - x_d) * x_d - 2) * x_d)**2

    return float(np.sum(term_1 + term_2))


def extended_trigonometric(x):
    cos_x = np.cos(x)

    term_1 = len(x) - np.sum(cos_x)
    term_2 = np.arange(1, len(x) + 1) * (1 - cos_x)
    term_3 = np.sin(x)

    return float(np.sum((term_1 + term_2 + term_3)**2))


def extended_rosenbrock(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]
    c = 100

    term_1 = c * (x_d - x_p**2)**2
    term_2 = (1 - x_p)**2

    return float(np.sum(term_1 + term_2))


def generalized_rosenbrock(x):
    x_from_1 = x[1:len(x)]
    x_from_0 = x[:len(x) - 1]
    c = 100

    term_1 = c * (x_from_1 - x_from_0**2)**2
    term_2 = (1 - x_from_0)**2

    return float(np.sum(term_1 + term_2))


def extended_white_and_holst(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]
    c = 100

    term_1 = c * (x_d - x_p**3)**2
    term_2 = (1 - x_p)**2

    return float(np.sum(term_1 + term_2))


def extended_baele(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]

    term_1 = (1.5 - x_p * (1 - x_d))**2
    term_2 = (2.25 - x_p * (1 - x_d**2))**2
    term_3 = (2.625 - x_p * (1 - x_d**3))**2

    return float(np.sum(term_1 + term_2 + term_3))


def extended_penalty(x):
    x_from_0 = x[:len(x) - 1]

    term_1 = (x_from_0 - 1)**2
    term_2 = (np.sum(x**2) - 0.25)**2

    return float(np.sum(term_1) + term_2)


def perturbed_quadratic(x):
    term_1 = power(x)
    term_2 = (1 / 100) * (np.sum(x))**2

    return float(term_1 + term_2)


def raydan_1(x):
    term = np.sum(np.arange(1, len(x) + 1) * (np.exp(x) - x))
    return float((1 / 10) * term)


def raydan_2(x):
    return float(np.sum(np.exp(x) - x))


def diagonal_1(x):
    return float(np.sum(np.exp(x) - np.arange(1, len(x) + 1) * x))


def diagonal_2(x):
    return float(np.sum(np.exp(x) - x / np.arange(1, len(x) + 1)))


def diagonal_3(x):
    return float(np.sum(np.exp(x) - np.arange(1, len(x) + 1) * np.sin(x)))


def hager(x):
    return float(np.sum(np.exp(x) - np.sqrt(np.arange(1, len(x) + 1)) * x))


def generalized_tridiagonal_1(x):
    x_from_1 = x[1:len(x)]
    x_from_0 = x[:len(x) - 1]

    term_1 = (x_from_0 + x_from_1 - 3)**2
    term_2 = (x_from_0 - x_from_1 + 1)**4

    return float(np.sum(term_1 + term_2))


def extended_tridiagonal_1(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]

    term_1 = (x_p + x_d - 3)**2
    term_2 = (x_p - x_d + 1)**4

    return float(np.sum(term_1 + term_2))


def diagonal_4(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]
    c = 100

    term = x_p**2 + c * x_d**2

    return float(0.5 * np.sum(term))


def diagonal_5(x):
    return float(np.sum(np.log(np.exp(x) + np.exp(-x))))


def extended_himmelblau(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]

    term_1 = (x_p**2 + x_d - 11)**2
    term_2 = (x_p + x_d**2 - 7)**2

    return float(np.sum(term_1 + term_2))


def generalized_white_and_holst(x):
    x_from_1 = x[1:len(x)]
    x_from_0 = x[:len(x) - 1]
    c = 100

    term_1 = c * (x_from_1 - x_from_0**3)**2
    term_2 = (1 - x_from_0)**2

    return float(np.sum(term_1 + term_2))


def extended_psc1(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]

    term_1 = (x_p**2 + x_d**2 + x_p * x_d)**2
    term_2 = (np.sin(x_p))**2
    term_3 = (np.cos(x_d))**2

    return float(np.sum(term_1 + term_2 + term_3))


def extended_bd1(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]

    term_1 = (x_p**2 + x_d - 2)**2
    term_2 = (np.exp(x_p - 1) - x_p)**2

    return float(np.sum(term_1 + term_2))


def extended_maratos(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]
    c = 100

    term_1 = x_p
    term_2 = c * (x_p**2 + x_d**2 - 1)**2

    return float(np.sum(term_1 + term_2))


def extended_cliff(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]

    term_1 = ((x_p - 3) / 100)**2
    term_2 = (x_p - x_d)
    term_3 = np.exp(20 * (x_p - x_d))

    return float(np.sum(term_1 + term_2 + term_3))


def perturbed_quadratic_diagonal(x):
    term_1 = (np.sum(x))**2
    term_2 = np.sum((np.arange(1, len(x) + 1) / 100) * x**2)

    return float(term_1 + term_2)


def extended_hiebert(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]

    term_1 = (x_p - 10)**2
    term_2 = (x_p * x_d - 50000)**2

    return float(np.sum(term_1 + term_2))


def quadratic_qf1(x):
    term_1 = 0.5 * power(x)
    term_2 = x[-1]

    return float(term_1 + term_2)


def extended_quadratic_penalty_qp1(x):
    term_1 = np.sum((x[:-1]**2 - 2)**2)
    term_2 = (np.sum(x**2) - 0.5)**2

    return float(term_1 + term_2)


def extended_quadratic_penalty_qp2(x):
    term_1 = np.sum((x[:-1]**2 - np.sin(x[:-1]))**2)
    term_2 = (np.sum(x**2) - 100)**2

    return float(term_1 + term_2)


def quadratic_qf2(x):
    term_1 = 0.5 * np.sum(np.arange(1, len(x) + 1) * (x**2 - 1)**2)
    term_2 = x[-1]

    return float(term_1 + term_2)


def extended_quadratic_exponential_ep1(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]

    term_1 = (np.exp(x_p - x_d) - 5)**2
    term_2 = (x_p - x_d)**2
    term_3 = (x_p - x_d - 11)**2

    return float(np.sum(term_1 + term_2 * term_3))


def extended_tridiagonal_2(x):
    x_from_1 = x[1:len(x)]
    x_from_0 = x[:len(x) - 1]
    c = 0.1

    term_1 = (x_from_1 * x_from_0 - 1)**2
    term_2 = c * (x_from_0 + 1) * (x_from_0 + 1)

    return float(np.sum(term_1 + term_2))


def fletcbv3(x):
    p = 10**(-8)
    h = 1 / (len(x) + 1)
    c = 1

    term_1 = 0.5 * p * (x[0]**2 + x[-1]**2)
    term_2 = np.sum((p / 2) * (x[:-1] - x[1:])**2)
    term_3 = np.sum((p * (h**2 + 2) / h**2) * x +
                    (c * p / h**2) * np.cos(x))

    return float(term_1 + term_2 + term_3)


def fletchcr(x):
    x_from_1 = x[1:len(x)]
    x_from_0 = x[:len(x) - 1]
    c = 100

    term = c * (x_from_1 - x_from_0 + 1 - x_from_0**2)**2

    return float(np.sum(term))


def bdqrtic(x):
    term_1 = (-4 * x[:-3] + 3)**2

    x = x**2
    x_minus_0 = x[np.arange(0, len(x) - 3)]
    x_minus_1 = x[np.arange(1, len(x) - 2)]
    x_minus_2 = x[np.arange(2, len(x) - 1)]
    x_minus_3 = x[np.arange(3, len(x))]

    term_2 = (x_minus_0 + 2 * x_minus_1 + 3 * x_minus_2 + 4 * x_minus_3 +
              5 * x[-1])**2

    return float(np.sum(term_1 + term_2))


def tridia(x):
    alfa = 2
    beta = 1
    gamma = 1
    delta = 1

    term_1 = gamma * (delta * x[0] - 1)**2
    term_2 = np.sum(
        np.arange(2,
                  len(x) + 1) * (alfa * x[1:] - beta * x[:-1])**2)

    return float(term_1 + term_2)


def arwhead(x):
    x_minus = x[:-1]

    term_1 = np.sum(-4 * x[:-1] + 3)

    x = x**2
    x_minus = x[:-1]

    term_2 = np.sum((x_minus + x[-1])**2)

    return float(term_1 + term_2)


def nondia(x):
    term_1 = (x[0] - 1)**2
    term_2 = 100 * np.sum(x[0] - x[1:]**2)**2

    return float(term_1 + term_2)


def nondquar(x):
    term_1 = (x[0] - x[1])**2
    term_2 = np.sum((x[:-2] + x[1:-1] + x[-1])**4)
    term_3 = (x[-2] + x[-1])**2

    return float(term_1 + term_2 + term_3)


def dqdrtic(x):
    x = x**2
    c = 100
    d = 100

    return float(np.sum(x[:-2] + c * x[1:-1] + d * x[2:]))


def eg2(x):
    x_0 = x[0]
    x = x**2

    term_1 = np.sum(np.sin(x_0 + x[:-1] - 1))
    term_2 = 0.5 * np.sin(x[-1])

    return float(term_1 + term_2)


def broyden_tridiagonal(x):
    x_sqr = x**2
    term_1 = (3 * x[0] - 2 * x_sqr[0])**2
    term_2 = np.sum(
        (3 * x[1:-1] - 2 * x_sqr[1:-1] - x[:-2] - 2 * x[2:] + 1)**2)
    term_3 = (3 * x[-1] - 2 * x_sqr[-1] - x[-2] + 1)**2

    return float(term_1 + term_2 + term_3)


def almost_perturbed_quadratic(x):
    term_1 = power(x)
    term_2 = (1 / 100) * (x[0] + x[-1])**2

    return float(term_1 + term_2)


def liarwhd(x):
    term_1 = 4 * np.sum((x**2 - x[0])**2)
    term_2 = np.sum((x - 1)**2)

    return float(term_1 + term_2)


def power(x):
    return float(np.sum(np.arange(1, len(x) + 1) * x**2))


def engval1(x):
    x_sqr = x**2

    term_1 = np.sum((x_sqr[:-1] + x_sqr[1:])**2)
    term_2 = np.sum(-4 * x[:-1] + 3)

    return float(term_1 + term_2)


def edensch(x):
    term_1 = (x[:-1] - 2)**4
    term_2 = (x[:-1] * x[1:] + 2 * x[1:])**2
    term_3 = (x[1:] + 1)**2

    return float(16 + np.sum(term_1 + term_2 + term_3))


def indef(x):
    term_1 = np.sum(x)
    term_2 = 0.5 * np.sum(np.cos(2 * x[1:-1] - x[-1] - x[0]))

    return float(term_1 + term_2)


def cube(x):
    term_1 = (x[0] - 1)**2
    term_2 = 100 * np.sum((x[1:] - x[:-1]**3)**2)

    return float(term_1 + term_2)


def bdexp(x):
    term_1 = x[:-2] + x[1:-1]
    term_2 = np.exp(-x[2:] * (term_1))

    return float(np.sum(term_1 * term_2))


def genhumps(x):
    term_1 = (np.sin(2 * x[:-1]))**2
    term_2 = (np.sin(2 * x[1:]))**2
    term_3 = 0.05 * (x[:-1]**2 + x[1:]**2)

    return float(np.sum(term_1 * term_2 + term_3))


def mccormck(x):
    term_1 = -1.5 * x[:-1] + 2.5 * x[1:] + 1
    term_2 = (x[:-1] - x[1:])**2
    term_3 = np.sin(x[:-1] + x[1:])

    return float(np.sum(term_1 + term_2 + term_3))


def nonscomp(x):
    term_1 = (x[0] - 1)**2
    term_2 = 4 * np.sum((x[1:] - x[:-1]**2)**2)

    return float(term_1 + term_2)


def vardim(x):
    n = len(x)

    term_1 = np.sum((x - 1)**2)
    term_2 = (np.sum(np.arange(1, n + 1) * x) - n * (n + 1) / 2)**2
    term_3 = (np.sum(np.arange(1, n + 1) * x) - n * (n + 1) / 2)**4

    return float(term_1 + term_2 + term_3)


def quartc(x):
    return float(np.sum((x - 1)**4))


def sinquad(x):
    x_sqr = x**2

    term_1 = (x[0] - 1)**4
    term_2 = np.sum((np.sin(x[1:-1] - x[-1]) - x_sqr[0] + x_sqr[1:-1])**2)
    term_3 = (x_sqr[-1] - x_sqr[0])**2

    return float(term_1 + term_2 + term_3)


def extended_denschnb(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]

    term_1 = (x_p - 2)**2
    term_2 = term_1 * x_d**2
    term_3 = (x_d + 1)**2

    return float(np.sum(term_1 + term_2 + term_3))


def extended_denschnf(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]

    term_1 = 2 * (x_p + x_d)**2
    term_2 = (x_p - x_d)**2 - 8
    term_3 = (5 * x_p**2 + (x_p - 3)**2 - 9)**2

    return float(np.sum((term_1 + term_2)**2 + term_3))


def dixon3dq(x):
    term_1 = (x[0] - 1)**2
    term_2 = np.sum((x[:-1] - x[1:])**2)
    term_3 = (x[-1] - 1)**2

    return float(term_1 + term_2 + term_3)


def cosine(x):
    return float(np.sum(np.cos(-0.5 * x[1:] + x[:-1]**2)))


def sine(x):
    return float(np.sum(np.sin(-0.5 * x[1:] + x[:-1]**2)))


def biggsb1(x):
    term_1 = (x[0] - 1)**2
    term_2 = np.sum((x[1:] - x[:-1])**2)
    term_3 = (x[-1] - 1)**2

    return float(term_1 + term_2 + term_3)


def generalized_quartic(x):
    term_1 = x[:-1]**2
    term_2 = (x[1:] + term_1)**2

    return float(np.sum(term_1 + term_2))


def diagonal_7(x):
    return float(np.sum(np.exp(x) - 2 * x - x**2))


def diagonal_8(x):
    return float(np.sum(x * np.exp(x) - 2 * x - x**2))


def fh3(x):
    return float((np.sum(x))**2 + diagonal_8(x))


def sincos(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]

    term_1 = (x_p**2 + x_d**2 + x_p * x_d)**2
    term_2 = (np.sin(x_p))**2
    term_3 = (np.cos(x_d))**2

    return float(np.sum(term_1 + term_2 + term_3))


def diagonal_9(x):
    return float(
        np.sum(np.exp(x) - np.arange(1,
                                     len(x) + 1) * x) + 10000 * x[-1]**2)


def himmelbg(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]

    term_1 = 2 * x_p**2 + 3 * x_d**2
    term_2 = np.exp(-x_p - x_d)

    return float(np.sum(term_1 * term_2))


def himmelh(x):
    x_p = x[:len(x) - 1:2]
    x_d = x[1:len(x):2]

    return float(np.sum(-3 * x_p - 2 * x_d + 2 + x_p**3 + x_d**2))


def ackley(x):
    a = 20
    b = 0.2
    c = 2 * math.pi

    term_1 = -a * np.exp(-b * np.sqrt((1 / len(x)) * np.sum(x**2)))
    term_2 = -np.exp((1 / len(x)) * np.sum(np.cos(c * x)))
    term_3 = a + np.exp(1)

    return float(term_1 + term_2 + term_3)


def griewank(x):
    term_1 = (1 / 4000) * np.sum(x**2)
    term_2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(term_1 - term_2 + 1)


def levy(x):
    w = 1 + (x - 1) / 4
    temp = w[:-1]
    w_d = w[-1]

    term_1 = np.sin(math.pi * w[0])**2
    term_2 = np.sum(
        (temp - 1)**2 * (1 + 10 * (np.sin(math.pi * temp + 1)**2)))
    term_3 = (w_d - 1)**2 * (1 + np.sin(2 * math.pi * w_d)**2)

    return float(term_1 + term_2 + term_3)


def rastrigin(x):
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * math.pi * x)))


def schwefel(x):
    return float(418.9829 * len(x) -
                 np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def sphere(x):
    return float(x.T @ x)


def sum_of_different_powers(x):
    return float(np.sum(np.abs(x)**np.arange(2, len(x) + 2)))


def trid(x):
    term_1 = np.sum((x - 1)**2)
    term_2 = np.sum(x[1:] * x[:-1])

    return float(term_1 + term_2)


def zakharov(x):
    term_1 = np.sum(x**2)
    term_2 = (0.5 * np.sum(np.arange(1, len(x) + 1) * x))**2
    term_3 = term_2**2

    return float(term_1 + term_2 + term_3)
