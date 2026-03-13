import numpy as np
import numpy.linalg as la
from functions import *
import math
import scipy as sp
from scipy.linalg import orth
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt


class ASGF:
    kind = 'Adaptive Stochastic Gradient-Free'

    data = {
        'm': 5,
        'A': 0.1,
        'B': 0.9,
        'A_minus': 0.95,
        'A_plus': 1.02,
        'B_minus': 0.98,
        'B_plus': 1.01,
        'gamma_L': 0.9,
        'gamma_sigma': 0.9,
        'r': 2,
        'ro': 0.01,
        'epsilon_m': 0.1,
        'threshold': 10**(-6),
        'sigma_zero': 0.01
    }

    def __init__(self, seed=2003, eps=1e-8, seed_env=0):
        self.seed = seed
        self.eps = eps
        self.seed_env = seed_env

    def optimize(self,
                 function,
                 dim=100,
                 it=1000,
                 x_init=None,
                 debug=True,
                 itprint=25):
        '''
        Principal method of the class.
        It optimizes the given function and returns the sequence of values found by the algorithm.
        
        Input:
        - function: function to optimize, string
        - dim: dimension of the domain of the function, int
        - it: number of iterations of the algorithm, int
        - x_init: initial point of the algorithm, array
        - debug: True if debug prints are wanted, bool
        - itprint: iterations to wait before mid-execution print, int
        - b: number of random directions for the smoothing, int

        Output:
        - list containing the best values found during the execution, in descending order
        - list containing all the values found during the execution
        - number of iterations performed
        '''
        np.random.seed(self.seed)
        f = Function(function, self.seed_env)

        if x_init is None:
            x = np.random.randn(dim)
        else:
            x = x_init

        steps = {}
        steps[0] = [x, f.evaluate(x)]

        best_value = steps[0][1]
        best_values = [[x, best_value]]

        norm = np.linalg.norm(x)
        ASGF.data['sigma_zero'] = norm / 10
        sigma = ASGF.data['sigma_zero']
        A = ASGF.data['A']
        B = ASGF.data['B']
        r = ASGF.data['r']
        L_nabla = 0
        lipschitz_coefficients = np.ones((dim, ))
        basis = special_ortho_group.rvs(dim)

        if debug:
            print('algorithm:', 'asgf', 'function:', function, 'dimension:',
                  len(x), 'initial value:', steps[0][1])

        for i in range(1, it + 1):
            try:
                if i % itprint == 0:
                    if debug:
                        print(i, 'th iteration - value:', steps[i - 1][1],
                              'last best value:', best_value)

                # grad = self.grad_estimator(x, f, b)
                grad, lipschitz_coefficients, lr, derivatives, L_nabla = self.grad_estimator(
                    x, ASGF.data['m'], sigma, len(x), lipschitz_coefficients,
                    basis, f, L_nabla, steps[i - 1][1])

                if not 'RLenvironment' in function:
                    x = x - lr * grad
                    steps[i] = [x, f.evaluate(x)]
                    if steps[i][1] < best_value:
                        best_value = steps[i][1]
                        best_values.append([steps[i][0], best_value])
                else:
                    x = x + lr * grad
                    steps[i] = [x, f.evaluate(x)]
                    if steps[i][1] > best_value:
                        best_value = steps[i][1]
                        best_values.append([steps[i][0], best_value])

                if la.norm(x - steps[i - 1][0]) < self.eps:
                    break
                else:
                    sigma, basis, A, B, r = self.subroutine(
                        sigma, grad, derivatives, lipschitz_coefficients, A, B,
                        r)

            except Exception as e:
                print('Something has gone wrong!')
                print(e)
                break

        if debug:
            print()
            try:
                print('last evaluation:', steps[i][1], 'last_iterate:', i,
                      'best evaluation:', best_value)
                print()
            except:
                print('last evaluation:', steps[i - 1][1], 'last_iterate:',
                      i - 1, 'best evaluation:', best_value)
                print()

        return best_values, [steps[j][1] for j in range(i)]

    def grad_estimator(self, x, m, sigma, dim, lipschitz_coefficients, basis,
                       f, L_nabla, value):
        '''
        Estimates the gradient and the learning rate.

        Input:
        - x: point on which the operation is performed, array
        - f: function over which the smoothing is performed, function object
        - m: number of points in the numerical approximation, int
        - sigma: smoothing parameter, float
        - dim: dimension of the domain, int
        - lipschitz_coefficients: list of lipschitz coefficients, array
        - basis: orthonormal basis for the domain, 2-D array
        - L_nabla: needed to compute learning rate, float
        - value: f(x), float

        Output:
        - the estimated gradient, array
        - list of updated lipschitz coefficients
        - learning rate
        - array of derivatives
        - updated L_nabla
        '''
        evaluations = {}

        points = {}
        derivatives = []
        p_5, w_5 = np.polynomial.hermite.hermgauss(m)
        p_w_5 = p_5 * w_5
        sigma_p_5 = sigma * p_5

        for i in range(dim):
            temp = []

            for k in range(m):
                if int(m / 2) == k:
                    evaluation = value
                else:
                    evaluation = f.evaluate(x + sigma_p_5[k] * basis[i])
                temp.append(evaluation)

            new_estimate = 2 / (sigma * np.sqrt(math.pi)) * np.sum(
                p_w_5 * np.array(temp))

            points[i] = p_5
            evaluations[i] = temp
            derivatives.append(new_estimate)

        # for i in range(dim):
        #     if i == 0:
        #         j = 3
        #         p, w = np.polynomial.hermite.hermgauss(j)
        #         sigma_p = sigma * p

        #         temp = []

        #         for k in range(j):
        #             evaluation = f(x + sigma_p[k] * basis[i])
        #             temp.append(evaluation)

        #         previous_estimate = 10000000000000
        #         new_estimate = 2 / (sigma * np.sqrt(math.pi)) * np.sum(
        #             w * p * np.array(temp))

        #         while np.abs(new_estimate -
        #                      previous_estimate) >= ASGF.data['epsilon_m']:
        #             j = j + 2

        #             p, w = np.polynomial.hermite.hermgauss(j)
        #             sigma_p = sigma * p

        #             temp = []

        #             for k in range(j):
        #                 evaluation = f(x + sigma_p[k] * basis[i])
        #                 temp.append(evaluation)

        #             previous_estimate = new_estimate
        #             new_estimate = 2 / (sigma * np.sqrt(math.pi)) * np.sum(
        #                 w * p * np.array(temp))

        #     else:
        #         temp = []
        #         p, w = p_5, w_5
        #         sigma_p = sigma_p_5

        #         for k in range(m):
        #             evaluation = f(x + sigma_p[k] * basis[i])
        #             temp.append(evaluation)

        #         new_estimate = 2 / (sigma * np.sqrt(math.pi)) * np.sum(
        #             p_w_5 * np.array(temp))

        #     points[i] = p
        #     evaluations[i] = temp
        #     derivatives.append(new_estimate)

        # grad = np.dot(basis.T, np.array(derivatives))
        grad = np.zeros(x.shape)

        for i in range(len(x)):
            grad = grad + derivatives[i] * basis[i]

        for i in range(len(grad)):
            temp = 0
            for k in range(len(points[i]) - 1):
                value = np.abs((evaluations[i][k + 1] - evaluations[i][k]) /
                               (sigma * (points[i][k + 1] - points[i][k])))

                if value > temp:
                    temp = value

            lipschitz_coefficients[i] = temp

        L_nabla = (1 - ASGF.data['gamma_L']) * lipschitz_coefficients[
            0] + ASGF.data['gamma_L'] * L_nabla

        lr = sigma / L_nabla

        return grad, lipschitz_coefficients, lr, np.array(derivatives), L_nabla

    def subroutine(self, sigma, grad, derivatives, lipschitz_coefficients, A,
                   B, r):
        '''
        Subroutine that updates the parameters of the algorithm to adapt it to the function.

        Input:
        - sigma: smoothing parameter, float
        - grad: estimated gradient, array
        - derivatives: estimated directional derivative, array
        - lipschitz_coefficients: list of lipschitz coefficients, array
        - A, B, r: parameters of the algorithm

        Output:
        - sigma: smoothing parameter, float
        - basis: orthonormal basis, 2-D array
        - A, B, r: updated parameters of the algorithm
        '''
        if r > 0 and sigma < ASGF.data['ro'] * ASGF.data['sigma_zero']:
            basis = sp.stats.special_ortho_group.rvs(len(grad))
            sigma = ASGF.data['sigma_zero']
            A = ASGF.data['A']
            B = ASGF.data['B']
            r = r - 1

            return sigma, basis, A, B, r

        else:
            basis = sp.stats.special_ortho_group.rvs(len(grad))
            basis[0] = grad / np.linalg.norm(grad)
            basis = orth(basis)

            value = np.max(np.abs(derivatives / lipschitz_coefficients))

            if value < A:
                sigma = sigma * ASGF.data['gamma_sigma']
                A = A * ASGF.data['A_minus']

                return sigma, basis, A, B, r

            elif value > B:
                sigma = sigma / ASGF.data['gamma_sigma']
                B = B * ASGF.data['B_plus']

                return sigma, basis, A, B, r

            else:
                A = A * ASGF.data['A_plus']
                B = B * ASGF.data['B_minus']

                return sigma, basis, A, B, r

    def plot_results(self, values, function):
        '''
        Plots the given values
        '''
        plt.plot(values, 'r', label=ASGF.kind)
        plt.yscale('log')
        plt.legend()
        plt.title(function)
        plt.show()
        pass