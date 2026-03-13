import numpy as np
import math
import gym
from scipy.special import expit


class Function:
    def __init__(self, name='sphere', seed_env=0):
        self.name = name
        self.seed_env = seed_env

    #  --------- Functions ---------

    def extended_feudenstein_and_roth(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]

        term_1 = (-13 + x_p + ((5 - x_d) * x_d - 2) * x_d)**2
        term_2 = (-29 + x_p + ((5 - x_d) * x_d - 2) * x_d)**2

        return float(np.sum(term_1 + term_2))

    def extended_trigonometric(self, x):
        cos_x = np.cos(x)

        term_1 = len(x) - np.sum(cos_x)
        term_2 = np.arange(1, len(x) + 1) * (1 - cos_x)
        term_3 = np.sin(x)

        return float(np.sum((term_1 + term_2 + term_3)**2))

    def extended_rosenbrock(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]
        c = 100

        term_1 = c * (x_d - x_p**2)**2
        term_2 = (1 - x_p)**2

        return float(np.sum(term_1 + term_2))

    def generalized_rosenbrock(self, x):
        x_from_1 = x[1:len(x)]
        x_from_0 = x[:len(x) - 1]
        c = 100

        term_1 = c * (x_from_1 - x_from_0**2)**2
        term_2 = (1 - x_from_0)**2

        return float(np.sum(term_1 + term_2))

    def extended_white_and_holst(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]
        c = 100

        term_1 = c * (x_d - x_p**3)**2
        term_2 = (1 - x_p)**2

        return float(np.sum(term_1 + term_2))

    def extended_baele(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]

        term_1 = (1.5 - x_p * (1 - x_d))**2
        term_2 = (2.25 - x_p * (1 - x_d**2))**2
        term_3 = (2.625 - x_p * (1 - x_d**3))**2

        return float(np.sum(term_1 + term_2 + term_3))

    def extended_penalty(self, x):
        x_from_0 = x[:len(x) - 1]

        term_1 = (x_from_0 - 1)**2
        term_2 = (np.sum(x**2) - 0.25)**2

        return float(np.sum(term_1) + term_2)

    def perturbed_quadratic(self, x):
        term_1 = self.power(x)
        term_2 = (1 / 100) * (np.sum(x))**2

        return float(term_1 + term_2)

    def raydan_1(self, x):
        term = np.sum(np.arange(1, len(x) + 1) * (np.exp(x) - x))
        return float((1 / 10) * term)

    def raydan_2(self, x):
        return float(np.sum(np.exp(x) - x))

    def diagonal_1(self, x):
        return float(np.sum(np.exp(x) - np.arange(1, len(x) + 1) * x))

    def diagonal_2(self, x):
        return float(np.sum(np.exp(x) - x / np.arange(1, len(x) + 1)))

    def diagonal_3(self, x):
        return float(np.sum(np.exp(x) - np.arange(1, len(x) + 1) * np.sin(x)))

    def hager(self, x):
        return float(np.sum(np.exp(x) - np.sqrt(np.arange(1, len(x) + 1)) * x))

    def generalized_tridiagonal_1(self, x):
        x_from_1 = x[1:len(x)]
        x_from_0 = x[:len(x) - 1]

        term_1 = (x_from_0 + x_from_1 - 3)**2
        term_2 = (x_from_0 - x_from_1 + 1)**4

        return float(np.sum(term_1 + term_2))

    def extended_tridiagonal_1(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]

        term_1 = (x_p + x_d - 3)**2
        term_2 = (x_p - x_d + 1)**4

        return float(np.sum(term_1 + term_2))

    def diagonal_4(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]
        c = 100

        term = x_p**2 + c * x_d**2

        return float(0.5 * np.sum(term))

    def diagonal_5(self, x):
        return float(np.sum(np.log(np.exp(x) + np.exp(-x))))

    def extended_himmelblau(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]

        term_1 = (x_p**2 + x_d - 11)**2
        term_2 = (x_p + x_d**2 - 7)**2

        return float(np.sum(term_1 + term_2))

    def generalized_white_and_holst(self, x):
        x_from_1 = x[1:len(x)]
        x_from_0 = x[:len(x) - 1]
        c = 100

        term_1 = c * (x_from_1 - x_from_0**3)**2
        term_2 = (1 - x_from_0)**2

        return float(np.sum(term_1 + term_2))

    def extended_psc1(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]

        term_1 = (x_p**2 + x_d**2 + x_p * x_d)**2
        term_2 = (np.sin(x_p))**2
        term_3 = (np.cos(x_d))**2

        return float(np.sum(term_1 + term_2 + term_3))

    def extended_bd1(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]

        term_1 = (x_p**2 + x_d - 2)**2
        term_2 = (np.exp(x_p - 1) - x_p)**2

        return float(np.sum(term_1 + term_2))

    def extended_maratos(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]
        c = 100

        term_1 = x_p
        term_2 = c * (x_p**2 + x_d**2 - 1)**2

        return float(np.sum(term_1 + term_2))

    def extended_cliff(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]

        term_1 = ((x_p - 3) / 100)**2
        term_2 = (x_p - x_d)
        term_3 = np.exp(20 * (x_p - x_d))

        return float(np.sum(term_1 + term_2 + term_3))

    def perturbed_quadratic_diagonal(self, x):
        term_1 = (np.sum(x))**2
        term_2 = np.sum((np.arange(1, len(x) + 1) / 100) * x**2)

        return float(term_1 + term_2)

    def extended_hiebert(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]

        term_1 = (x_p - 10)**2
        term_2 = (x_p * x_d - 50000)**2

        return float(np.sum(term_1 + term_2))

    def quadratic_qf1(self, x):
        term_1 = 0.5 * self.power(x)
        term_2 = x[-1]

        return float(term_1 + term_2)

    def extended_quadratic_penalty_qp1(self, x):
        term_1 = np.sum((x[:-1]**2 - 2)**2)
        term_2 = (np.sum(x**2) - 0.5)**2

        return float(term_1 + term_2)

    def extended_quadratic_penalty_qp2(self, x):
        term_1 = np.sum((x[:-1]**2 - np.sin(x[:-1]))**2)
        term_2 = (np.sum(x**2) - 100)**2

        return float(term_1 + term_2)

    def quadratic_qf2(self, x):
        term_1 = 0.5 * np.sum(np.arange(1, len(x) + 1) * (x**2 - 1)**2)
        term_2 = x[-1]

        return float(term_1 + term_2)

    def extended_quadratic_exponential_ep1(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]

        term_1 = (np.exp(x_p - x_d) - 5)**2
        term_2 = (x_p - x_d)**2
        term_3 = (x_p - x_d - 11)**2

        return float(np.sum(term_1 + term_2 * term_3))

    def extended_tridiagonal_2(self, x):
        x_from_1 = x[1:len(x)]
        x_from_0 = x[:len(x) - 1]
        c = 0.1

        term_1 = (x_from_1 * x_from_0 - 1)**2
        term_2 = c * (x_from_0 + 1) * (x_from_0 + 1)

        return float(np.sum(term_1 + term_2))

    def fletcbv3(self, x):
        p = 10**(-8)
        h = 1 / (len(x) + 1)
        c = 1

        term_1 = 0.5 * p * (x[0]**2 + x[-1]**2)
        term_2 = np.sum((p / 2) * (x[:-1] - x[1:])**2)
        term_3 = np.sum((p * (h**2 + 2) / h**2) * x +
                        (c * p / h**2) * np.cos(x))

        return float(term_1 + term_2 + term_3)

    def fletchcr(self, x):
        x_from_1 = x[1:len(x)]
        x_from_0 = x[:len(x) - 1]
        c = 100

        term = c * (x_from_1 - x_from_0 + 1 - x_from_0**2)**2

        return float(np.sum(term))

    def bdqrtic(self, x):
        term_1 = (-4 * x[:-3] + 3)**2

        x = x**2
        x_minus_0 = x[np.arange(0, len(x) - 3)]
        x_minus_1 = x[np.arange(1, len(x) - 2)]
        x_minus_2 = x[np.arange(2, len(x) - 1)]
        x_minus_3 = x[np.arange(3, len(x))]

        term_2 = (x_minus_0 + 2 * x_minus_1 + 3 * x_minus_2 + 4 * x_minus_3 +
                  5 * x[-1])**2

        return float(np.sum(term_1 + term_2))

    def tridia(self, x):
        alfa = 2
        beta = 1
        gamma = 1
        delta = 1

        term_1 = gamma * (delta * x[0] - 1)**2
        term_2 = np.sum(
            np.arange(2,
                      len(x) + 1) * (alfa * x[1:] - beta * x[:-1])**2)

        return float(term_1 + term_2)

    def arwhead(self, x):
        x_minus = x[:-1]

        term_1 = np.sum(-4 * x[:-1] + 3)

        x = x**2
        x_minus = x[:-1]

        term_2 = np.sum((x_minus + x[-1])**2)

        return float(term_1 + term_2)

    def nondia(self, x):
        term_1 = (x[0] - 1)**2
        term_2 = 100 * np.sum(x[0] - x[1:]**2)**2

        return float(term_1 + term_2)

    def nondquar(self, x):
        term_1 = (x[0] - x[1])**2
        term_2 = np.sum((x[:-2] + x[1:-1] + x[-1])**4)
        term_3 = (x[-2] + x[-1])**2

        return float(term_1 + term_2 + term_3)

    def dqdrtic(self, x):
        x = x**2
        c = 100
        d = 100

        return float(np.sum(x[:-2] + c * x[1:-1] + d * x[2:]))

    def eg2(self, x):
        x_0 = x[0]
        x = x**2

        term_1 = np.sum(np.sin(x_0 + x[:-1] - 1))
        term_2 = 0.5 * np.sin(x[-1])

        return float(term_1 + term_2)

    def broyden_tridiagonal(self, x):
        x_sqr = x**2
        term_1 = (3 * x[0] - 2 * x_sqr[0])**2
        term_2 = np.sum(
            (3 * x[1:-1] - 2 * x_sqr[1:-1] - x[:-2] - 2 * x[2:] + 1)**2)
        term_3 = (3 * x[-1] - 2 * x_sqr[-1] - x[-2] + 1)**2

        return float(term_1 + term_2 + term_3)

    def almost_perturbed_quadratic(self, x):
        term_1 = self.power(x)
        term_2 = (1 / 100) * (x[0] + x[-1])**2

        return float(term_1 + term_2)

    def liarwhd(self, x):
        term_1 = 4 * np.sum(-x[0] + x**2)
        term_2 = np.sum((x - 1)**2)

        return float(term_1 + term_2)

    def power(self, x):
        return float(np.sum(np.arange(1, len(x) + 1) * x**2))

    def engval1(self, x):
        x_sqr = x**2

        term_1 = np.sum((x_sqr[:-1] + x_sqr[1:])**2)
        term_2 = np.sum(-4 * x[:-1] + 3)

        return float(term_1 + term_2)

    def edensch(self, x):
        term_1 = (x[:-1] - 2)**4
        term_2 = (x[:-1] * x[1:] + 2 * x[1:])**2
        term_3 = (x[1:] + 1)**2

        return float(16 + np.sum(term_1 + term_2 + term_3))

    def indef(self, x):
        term_1 = np.sum(x)
        term_2 = 0.5 * np.sum(np.cos(2 * x[1:-1] - x[-1] - x[0]))

        return float(term_1 + term_2)

    def cube(self, x):
        term_1 = (x[0] - 1)**2
        term_2 = 100 * np.sum((x[1:] - x[:-1]**3)**2)

        return float(term_1 + term_2)

    def bdexp(self, x):
        term_1 = x[:-2] + x[1:-1]
        term_2 = np.exp(-x[2:] * (term_1))

        return float(np.sum(term_1 * term_2))

    def genhumps(self, x):
        term_1 = (np.sin(2 * x[:-1]))**2
        term_2 = (np.sin(2 * x[1:]))**2
        term_3 = 0.05 * (x[:-1]**2 + x[1:]**2)

        return float(np.sum(term_1 * term_2 + term_3))

    def mccormck(self, x):
        term_1 = -1.5 * x[:-1] + 2.5 * x[1:] + 1
        term_2 = (x[:-1] - x[1:])**2
        term_3 = np.sin(x[:-1] + x[1:])

        return float(np.sum(term_1 + term_2 + term_3))

    def nonscomp(self, x):
        term_1 = (x[0] - 1)**2
        term_2 = 4 * np.sum((x[1:] - x[:-1]**2)**2)

        return float(term_1 + term_2)

    def vardim(self, x):
        n = len(x)

        term_1 = np.sum((x - 1)**2)
        term_2 = (np.sum(np.arange(1, n + 1) * x) - n * (n + 1) / 2)**2
        term_3 = (np.sum(np.arange(1, n + 1) * x) - n * (n + 1) / 2)**4

        return float(term_1 + term_2 + term_3)

    def quartc(self, x):
        return float(np.sum((x - 1)**4))

    def sinquad(self, x):
        x_sqr = x**2

        term_1 = (x[0] - 1)**4
        term_2 = np.sum((np.sin(x[1:-1] - x[-1]) - x_sqr[0] + x_sqr[1:-1])**2)
        term_3 = (x_sqr[-1] - x_sqr[0])**2

        return float(term_1 + term_2 + term_3)

    def extended_denschnb(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]

        term_1 = (x_p - 2)**2
        term_2 = term_1 * x_d**2
        term_3 = (x_d + 1)**2

        return float(np.sum(term_1 + term_2 + term_3))

    def extended_denschnf(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]

        term_1 = 2 * (x_p + x_d)**2
        term_2 = (x_p - x_d)**2 - 8
        term_3 = (5 * x_p**2 + (x_p - 3)**2 - 9)**2

        return float(np.sum((term_1 + term_2)**2 + term_3))

    def liarwhd(self, x):
        term_1 = 4 * np.sum((x**2 - x[0])**2)
        term_2 = np.sum((x - 1)**2)

        return float(term_1 + term_2)

    def dixon3dq(self, x):
        term_1 = (x[0] - 1)**2
        term_2 = np.sum((x[:-1] - x[1:])**2)
        term_3 = (x[-1] - 1)**2

        return float(term_1 + term_2 + term_3)

    def cosine(self, x):
        return float(np.sum(np.cos(-0.5 * x[1:] + x[:-1]**2)))

    def sine(self, x):
        return float(np.sum(np.sin(-0.5 * x[1:] + x[:-1]**2)))

    def biggsb1(self, x):
        term_1 = (x[0] - 1)**2
        term_2 = np.sum((x[1:] - x[:-1])**2)
        term_3 = (x[-1] - 1)**2

        return float(term_1 + term_2 + term_3)

    def generalized_quartic(self, x):
        term_1 = x[:-1]**2
        term_2 = (x[1:] + term_1)**2

        return float(np.sum(term_1 + term_2))

    def diagonal_7(self, x):
        return float(np.sum(np.exp(x) - 2 * x - x**2))

    def diagonal_8(self, x):
        return float(np.sum(x * np.exp(x) - 2 * x - x**2))

    def fh3(self, x):
        return float((np.sum(x))**2 + self.diagonal_8(x))

    def sincos(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]

        term_1 = (x_p**2 + x_d**2 + x_p * x_d)**2
        term_2 = (np.sin(x_p))**2
        term_3 = (np.cos(x_d))**2

        return float(np.sum(term_1 + term_2 + term_3))

    def diagonal_9(self, x):
        return float(
            np.sum(np.exp(x) - np.arange(1,
                                         len(x) + 1) * x) + 10000 * x[-1]**2)

    def himmelbg(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]

        term_1 = 2 * x_p**2 + 3 * x_d**2
        term_2 = np.exp(-x_p - x_d)

        return float(np.sum(term_1 * term_2))

    def himmelh(self, x):
        x_p = x[:len(x) - 1:2]
        x_d = x[1:len(x):2]

        return float(np.sum(-3 * x_p - 2 * x_d + 2 + x_p**3 + x_d**2))

    def ackley(self, x):
        a = 20
        b = 0.2
        c = 2 * math.pi

        term_1 = -a * np.exp(-b * np.sqrt((1 / len(x)) * np.sum(x**2)))
        term_2 = -np.exp((1 / len(x)) * np.sum(np.cos(c * x)))
        term_3 = a + np.exp(1)

        return float(term_1 + term_2 + term_3)

    def griewank(self, x):
        term_1 = (1 / 4000) * np.sum(x**2)
        term_2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return float(term_1 - term_2 + 1)

    def levy(self, x):
        w = 1 + (x - 1) / 4
        temp = w[:-1]
        w_d = w[-1]

        term_1 = np.sin(math.pi * w[0])**2
        term_2 = np.sum(
            (temp - 1)**2 * (1 + 10 * (np.sin(math.pi * temp + 1)**2)))
        term_3 = (w_d - 1)**2 * (1 + np.sin(2 * math.pi * w_d)**2)

        return float(term_1 + term_2 + term_3)

    def rastrigin(self, x):
        return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * math.pi * x)))

    def schwefel(self, x):
        return float(418.9829 * len(x) -
                     np.sum(x * np.sin(np.sqrt(np.abs(x)))))

    def sphere(self, x):
        return float(x.T @ x)

    def sum_of_different_powers(self, x):
        return float(np.sum(np.abs(x)**np.arange(2, len(x) + 2)))

    def trid(self, x):
        term_1 = np.sum((x - 1)**2)
        term_2 = np.sum(x[1:] * x[:-1])

        return float(term_1 + term_2)

    def zakharov(self, x):
        term_1 = np.sum(x**2)
        term_2 = (0.5 * np.sum(np.arange(1, len(x) + 1) * x))**2
        term_3 = term_2**2

        return float(term_1 + term_2 + term_3)

    def relu(self, x):
        return abs(x) * (x > 0)

    def softmax(self, x):
        y = np.exp(x - np.max(x))
        f_x = y / np.sum(np.exp(x))
        return f_x

    def RLenvironmentPendulum(self, x):
        env = gym.make('Pendulum-v0')
        env.seed(self.seed_env)
        observation = env.reset()
        h = 5
        obs_space = 3
        act_space = 1
        rewards = []
        for t in range(200):
            W_1 = x[:obs_space * h].reshape((obs_space, h))
            W_2 = x[obs_space * h:].reshape((act_space, h))
            action = expit(
                np.dot(
                    W_2,
                    self.relu(np.dot(W_1.T, observation.reshape((obs_space, 1))))))

            observation, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                break
        env.close()
        return np.sum(rewards)

    def RLenvironmentCartPole(self, x):
        env = gym.make('CartPole-v1')
        env.seed(self.seed_env)
        observation = env.reset()
        h = 4
        obs_space = 4
        act_space = 1
        rewards = []
        for t in range(200):
            W_1 = x[:obs_space * h].reshape((obs_space, h))
            W_2 = x[obs_space * h:].reshape((act_space, h))
            action = np.dot(
                W_2, self.relu(np.dot(W_1.T, observation.reshape((obs_space, 1)))))
            if action > 0:
                action = 1
            else:
                action = 0
            observation, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                break
        env.close()
        return np.sum(rewards)

    functions_list = {
        'extended_feudenstein_and_roth': extended_feudenstein_and_roth,
        'extended_trigonometric': extended_trigonometric,
        'extended_rosenbrock': extended_rosenbrock,
        'generalized_rosenbrock': generalized_rosenbrock,
        'extended_white_and_holst': extended_white_and_holst,
        'extended_baele': extended_baele,
        'extended_penalty': extended_penalty,
        'perturbed_quadratic': perturbed_quadratic,
        'raydan_1': raydan_1,
        'raydan_2': raydan_2,
        'diagonal_1': diagonal_1,
        'diagonal_2': diagonal_2,
        'diagonal_3': diagonal_3,
        'hager': hager,
        'extended_tridiagonal_1': extended_tridiagonal_1,
        'diagonal_4': diagonal_4,
        'diagonal_5': diagonal_5,
        'extended_himmelblau': extended_himmelblau,
        'generalized_white_and_holst': generalized_white_and_holst,
        'extended_psc1': extended_psc1,
        'extended_bd1': extended_bd1,
        'extended_maratos': extended_maratos,
        'extended_cliff': extended_cliff,
        'perturbed_quadratic_diagonal': perturbed_quadratic_diagonal,
        'extended_hiebert': extended_hiebert,
        'quadratic_qf1': quadratic_qf1,
        'extended_quadratic_penalty_qp1': extended_quadratic_penalty_qp1,
        'extended_quadratic_penalty_qp2': extended_quadratic_penalty_qp2,
        'quadratic_qf2': quadratic_qf2,
        'extended_quadratic_exponential_ep1':
        extended_quadratic_exponential_ep1,
        'extended_tridiagonal_2': extended_tridiagonal_2,
        'fletcbv3': fletcbv3,
        'fletchcr': fletchcr,
        'bdqrtic': bdqrtic,
        'tridia': tridia,
        'arwhead': arwhead,
        'nondia': nondia,
        'nondquar': nondquar,
        'dqdrtic': dqdrtic,
        'eg2': eg2,
        'broyden_tridiagonal': broyden_tridiagonal,
        'almost_perturbed_quadratic': almost_perturbed_quadratic,
        'liarwhd': liarwhd,
        'power': power,
        'engval1': engval1,
        'edensch': edensch,
        'indef': indef,
        'cube': cube,
        'bdexp': bdexp,
        'genhumps': genhumps,
        'mccormck': mccormck,
        'nonscomp': nonscomp,
        'vardim': vardim,
        'quartc': quartc,
        'sinquad': sinquad,
        'extended_denschnb': extended_denschnb,
        'extended_denschnf': extended_denschnf,
        'liarwhd': liarwhd,
        'dixon3dq': dixon3dq,
        'cosine': cosine,
        'sine': sine,
        'biggsb1': biggsb1,
        'generalized_quartic': generalized_quartic,
        'diagonal_7': diagonal_7,
        'diagonal_8': diagonal_8,
        'fh3': fh3,
        'sincos': sincos,
        'diagonal_9': diagonal_9,
        'himmelbg': himmelbg,
        'himmelh': himmelh,
        'ackley': ackley,
        'griewank': griewank,
        'levy': levy,
        'rastrigin': rastrigin,
        'schwefel': schwefel,
        'sphere': sphere,
        'sum_of_different_powers': sum_of_different_powers,
        'trid': trid,
        'zakharov': zakharov,
        'RLenvironmentPendulum': RLenvironmentPendulum,
        'RLenvironmentCartPole': RLenvironmentCartPole
    }

    # -------- Methods --------

    def evaluate(self, x):
        return self.functions_list[self.name](self, x)