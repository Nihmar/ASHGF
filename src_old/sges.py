import numpy as np
import numpy.linalg as la
from functions import *
import matplotlib.pyplot as plt


class SGES:
    kind = 'Self-Guided Evolution Strategies'

    def __init__(self,
                 lr=1e-4,
                 sigma=1e-4,
                 k=50,
                 k1=0.9,
                 k2=0.1,
                 alpha=0.5,
                 delta=1.1,
                 t=50,
                 seed=2003,
                 eps=1e-8,
                 seed_env=0):
        if lr < 0:
            raise ValueError("Error: learning rate < 0")
        if sigma < 0:
            raise ValueError("Error: sigma < 0")

        self.lr = lr
        self.sigma = sigma
        self.k = k
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

        G = []

        if debug:
            print('algorithm:', 'sges', 'function:', function, 'dimension:',
                  len(x), 'initial value:', steps[0][1])

        for i in range(1, it + 1):
            try:
                if i % itprint == 0:
                    if debug:
                        print(i, 'th iteration - value:', steps[i - 1][1],
                              'last best value:', best_value)

                if i < self.t:
                    grad = self.grad_estimator(x, f)
                    G.append(grad)
                else:
                    grad, evaluations, M = self.grad_estimator(
                        x, f, G, True, alpha)
                    G.append(grad)
                    G = G[1:]

                    try:
                        r = (1 / M) * np.sum([
                            min([evaluations[2 * j], evaluations[2 * j + 1]])
                            for j in range(M)
                        ])
                    except:
                        r = False

                    try:
                        r_hat = (1 / (dim - M)) * np.sum([
                            min([evaluations[2 * j], evaluations[2 * j + 1]])
                            for j in range(M, dim)
                        ])
                    except:
                        r_hat = False

                    if not r or r < r_hat:
                        alpha = min([self.delta * alpha, self.k1])
                    elif not r_hat or r >= r_hat:
                        alpha = max([(1 / self.delta) * alpha, self.k2])
                    else:
                        pass

                if not 'RLenvironment' in function:
                    x = x - self.lr * grad
                    steps[i] = [x, f.evaluate(x)]
                    if steps[i][1] < best_value:
                        best_value = steps[i][1]
                        best_values.append([steps[i][0], best_value])
                else:
                    x = x + self.lr * grad
                    steps[i] = [x, f.evaluate(x)]
                    if steps[i][1] > best_value:
                        best_value = steps[i][1]
                        best_values.append([steps[i][0], best_value])

                if la.norm(x - steps[i - 1][0]) < self.eps:
                    break

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

    def grad_estimator(self, x, f, G=None, sges=False, alpha=0):
        '''
        Performs the Central Gaussian Smoothing around x for the function f.

        Input:
        - x: point on which the operation is performed, array
        - f: function over which the smoothing is performed, function object
        - G: buffer of gradients, 2-D array
        - sges: True if to compute special directions of SGES, bool
        - alpha: probability of sampling from gradients, float

        Output:
        - the estimated gradient, array
        '''
        np.random.seed(self.seed)

        dim = len(x)

        grad = np.zeros(dim, )

        if sges:
            directions, M = self.compute_directions_sges(dim, G, alpha)
        directions = self.compute_directions(dim)

        evaluations = []

        for i in range(dim):
            d = directions[i].reshape(x.shape)
            dir_plus = x + self.sigma * d
            dir_minus = x - self.sigma * d

            evaluations_plus = f.evaluate(dir_plus)
            evaluations_minus = f.evaluate(dir_minus)

            evaluations.append(evaluations_plus)
            evaluations.append(evaluations_minus)

            grad = grad + (evaluations_plus - evaluations_minus) * d.reshape(
                grad.shape)

        grad = grad / (2 * self.sigma * dim)

        if sges:
            return grad, evaluations, M
        else:
            return grad

    def compute_directions(self, dim):
        '''
        Compute a matrix of random directions.

        Input:
        - dim: number of rows and columns, int

        Output:
        - matrix of directions
        '''
        return np.random.randn(dim, dim)

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
        - covariance of the directions
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

        # return dirs, choices, alpha * cov_L_G + (1 - alpha) * np.identity(dim)

    def plot_results(self, values, function):
        '''
        Plots the given values
        '''
        plt.plot(values, 'r', label=SGES.kind)
        plt.yscale('log')
        plt.legend()
        plt.title(function)
        plt.show()
        pass