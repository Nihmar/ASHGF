import numpy as np
import numpy.linalg as la
from functions import *
import math
import scipy as sp
from scipy.linalg import orth
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt


class ASHGF:
    kind = 'Adaptive Stochastic Historical Gradient-Free'

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
        'gamma_sigma_plus': 1 / 0.9,
        'gamma_sigma_minus': 0.9,
        'r': 10,
        'ro': 0.01,
        'threshold': 10**(-6),
        'sigma_zero': 0.01
    }

    def __init__(self,
                 k1=0.9,
                 k2=0.1,
                 alpha=0.5,
                 delta=1.1,
                 t=50,
                 seed=2003,
                 eps=1e-8,
                 seed_env=0):
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha
        self.delta = delta
        self.t = t
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

        Output:
        - list containing the best values found during the execution, in descending order
        - list containing all the values found during the execution
        - number of iterations performed
        '''
        np.random.seed(self.seed)
        f = Function(function, self.seed_env)
        alpha = self.alpha

        if x_init is None:
            x = np.random.randn(dim)
        else:
            x = x_init

        steps = {}
        steps[0] = [x, f.evaluate(x)]

        best_value = steps[0][1]
        best_values = [[x, best_value]]

        norm = np.linalg.norm(x)
        ASHGF.data['sigma_zero'] = norm / 10
        sigma = ASHGF.data['sigma_zero']
        A = ASHGF.data['A']
        B = ASHGF.data['B']
        r = ASHGF.data['r']
        L_nabla = 0
        M = dim
        lipschitz_coefficients = np.ones((dim, ))
        basis = special_ortho_group.rvs(dim)

        G = []

        if debug:
            print('algorithm:', 'ashgf', 'function:', function, 'dimension:',
                  len(x), 'initial value:', steps[0][1])

        for i in range(1, it + 1):
            try:
                if i % itprint == 0:
                    if debug:
                        print(i, 'th iteration - value:', steps[i - 1][1],
                              'last best value:', best_value)

                # grad = self.grad_estimator(x, f, b)
                grad, lipschitz_coefficients, lr, derivatives, L_nabla, evaluations = self.grad_estimator(
                    x, ASHGF.data['m'], sigma, len(x), lipschitz_coefficients,
                    basis, f, L_nabla, M, steps[i - 1][1])

                G.append(grad)
                if len(G) > self.t:
                    G = G[1:]

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
                    if i < self.t:
                        historical = False

                    else:
                        if i >= self.t + 1:
                            try:
                                r_ = (1 / M) * np.sum(
                                    [min(evaluations[j]) for j in range(M)])
                            except:
                                r_ = False

                            try:
                                r_hat = (1 / (dim - M)) * np.sum([
                                    min(evaluations[j]) for j in range(M, dim)
                                ])
                            except:
                                r_hat = False

                            if not r_ or r_ < r_hat:
                                alpha = min([self.delta * alpha, self.k1])
                            elif not r_hat or r_ >= r_hat:
                                alpha = max([(1 / self.delta) * alpha,
                                             self.k2])
                            else:
                                pass

                        historical = True

                    sigma, basis, A, B, r, M = self.subroutine(
                        sigma, grad, derivatives, lipschitz_coefficients, A, B,
                        r, G, alpha, historical)

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
                       f, L_nabla, M, value):
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
        - M: number of directions sampled from gradient subspace, int
        - value: f(x), float

        Output:
        - the estimated gradient, array
        - list of updated lipschitz coefficients
        - learning rate
        - array of derivatives
        - updated L_nabla
        - the evaluations of the function during the estimation
        '''
        evaluations = {}

        points = {}
        derivatives = []
        p_5, w_5 = np.polynomial.hermite.hermgauss(m)
        p_w_5 = p_5 * w_5
        sigma_p_5 = sigma * p_5

        buffer = []
        n = m

        for i in range(n):
            for j in range(n):
                if [i, j] or [j, i] not in buffer:
                    if np.abs(i - int(m / 2)) != np.abs(j - int(m / 2)):
                        buffer.append([i, j])

        for i in range(dim):
            temp = []

            for k in range(m):
                try:
                    if int(m / 2) == k:
                        evaluation = value
                    else:
                        evaluation = f.evaluate(x + sigma_p_5[k] * basis[i])
                    temp.append(evaluation)
                except:
                    print(x.shape, basis.shape)
                    sys.exit(1)

            new_estimate = 2 / (sigma * np.sqrt(math.pi)) * np.sum(
                p_w_5 * np.array(temp))

            points[i] = p_5
            evaluations[i] = temp
            derivatives.append(new_estimate)

        # grad = np.dot(basis.T, np.array(derivatives))
        grad = np.zeros(x.shape)

        for i in range(len(x)):
            grad = grad + derivatives[i] * basis[i]

        for i in range(len(grad)):
            temp = 0
            for couple in buffer:
                value = np.abs(
                    (evaluations[i][couple[0]] - evaluations[i][couple[1]]) /
                    (sigma * (points[i][couple[0]] - points[i][couple[1]])))

                if value > temp:
                    temp = value

            lipschitz_coefficients[i] = temp

        L_nabla = (1 - ASHGF.data['gamma_L']) * np.max(
            lipschitz_coefficients[:M]) + ASHGF.data['gamma_L'] * L_nabla

        lr = sigma / L_nabla

        return grad, lipschitz_coefficients, lr, np.array(
            derivatives), L_nabla, evaluations

    def subroutine(self, sigma, grad, derivatives, lipschitz_coefficients, A,
                   B, r, G, alpha, historical):
        '''
        Subroutine that updates the parameters of the algorithm to adapt it to the function.

        Input:
        - sigma: smoothing parameter, float
        - grad: estimated gradient, array
        - derivatives: estimated directional derivative, array
        - lipschitz_coefficients: list of lipschitz coefficients, array
        - A, B, r: parameters of the algorithm
        - G: buffer of gradients, 2-D array
        - alpha: probability of sampling from gradient subspace, float
        - historical: if true, use G, bool

        Output:
        - sigma: smoothing parameter, float
        - basis: orthonormal basis, 2-D array
        - A, B, r: updated parameters of the algorithm
        '''
        if r > 0 and sigma < ASHGF.data['ro'] * ASHGF.data['sigma_zero']:
            basis = sp.stats.special_ortho_group.rvs(len(grad))
            sigma = ASHGF.data['sigma_zero']
            A = ASHGF.data['A']
            B = ASHGF.data['B']
            r = r - 1
            M = int(len(grad) / 2)
            return sigma, basis, A, B, r, M

        else:
            if historical:
                basis, M = self.compute_directions_sges(len(grad), G, alpha)
                basis = orth(basis)
            else:
                M = int(len(grad) / 2)
                basis = sp.stats.special_ortho_group.rvs(len(grad))

            while basis.shape != (len(grad), len(grad)):
                v = np.random.randn(len(grad), len(grad) - basis.shape[1])
                basis = np.concatenate((basis.T, v.T))
                basis = orth(basis)

            value = np.max(np.abs(derivatives / lipschitz_coefficients))

            if value < A:
                sigma = sigma * ASHGF.data['gamma_sigma_minus']
                A = A * ASHGF.data['A_minus']

                return sigma, basis, A, B, r, M

            elif value > B:
                sigma = sigma * ASHGF.data['gamma_sigma_plus']
                B = B * ASHGF.data['B_plus']

                return sigma, basis, A, B, r, M

            else:
                A = A * ASHGF.data['A_plus']
                B = B * ASHGF.data['B_minus']

                return sigma, basis, A, B, r, M

    def compute_directions_sges(self, dim, G, alpha):
        '''
        Compute a matrix of random directions depending on gradient information.

        Input:
        - dim: number of rows and columns, int
        - G: buffer of gradients, 2-D array
        - alpha: probabilità of sampling from gradients, float

        Output:
        - matrix of directions
        - number of directions sampled from gradients
        '''
        G = np.array(G)
        cov_L_G = np.cov(G.T)

        choices = 0

        for i in range(dim):
            choices += int(
                np.random.choice([0, 1], size=1, p=[alpha, 1 - alpha]))

        dirs_L_G = np.random.multivariate_normal(np.zeros(dim), cov_L_G,
                                                 choices)
        for i in range(choices):
            dirs_L_G[i] = dirs_L_G[i] / np.std(dirs_L_G[i])

        dirs_L_G_T = np.random.multivariate_normal(np.zeros(dim),
                                                   np.identity(dim),
                                                   dim - choices)

        dirs = np.concatenate((dirs_L_G, dirs_L_G_T))

        return dirs / np.linalg.norm(dirs, axis=-1), choices

    def plot_results(self, values, function):
        '''
        Plots the given values
        '''
        plt.plot(values, 'r', label=ASHGF.kind)
        plt.yscale('log')
        plt.legend()
        plt.title(function)
        plt.show()
        pass