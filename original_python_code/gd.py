import numpy as np
import numpy.linalg as la
from functions import *
import matplotlib.pyplot as plt


class GD:
    kind = 'Vanilla Gradient Descent'

    def __init__(self, lr=1e-4, sigma=1e-4, seed=2003, eps=1e-8, seed_env=0):
        if lr < 0:
            raise ValueError("Error: learning rate < 0")
        if sigma < 0:
            raise ValueError("Error: sigma < 0")

        self.lr = lr
        self.sigma = sigma
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

        if x_init is None:
            x = np.random.randn(dim)
        else:
            x = x_init

        steps = {}
        steps[0] = [x, f.evaluate(x)]

        best_value = steps[0][1]
        best_values = [[x, best_value]]

        if debug:
            print('algorithm:', 'gd', 'function:', function, 'dimension:',
                  len(x), 'initial value:', steps[0][1])

        for i in range(1, it + 1):
            try:
                if i % itprint == 0:
                    if debug:
                        print(i, 'th iteration - value:', steps[i - 1][1],
                              'last best value:', best_value)

                grad = self.grad_estimator(x, f)

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

    def grad_estimator(self, x, f):
        '''
        Performs the Central Gaussian Smoothing around x for the function f.

        Input:
        - x: point on which the operation is performed, array
        - f: function over which the smoothing is performed, function object

        Output:
        - the estimated gradient, array
        '''
        dim = len(x)

        grad = np.zeros(dim, )
        directions = self.compute_directions(dim, dim)

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

        return grad

    def compute_directions(self, dim_1, dim_2):
        '''
        Compute a matrix of random directions.

        Input:
        - dim_1: number of rows, int
        - dim_2: number of cols, int

        Output:
        - matrix of directions
        '''
        return np.random.randn(dim_1, dim_2)

    def plot_results(self, values, function):
        '''
        Plots the given values
        '''
        plt.plot(values, 'r', label=GD.kind)
        plt.yscale('log')
        plt.legend()
        plt.title(function)
        plt.show()
        pass