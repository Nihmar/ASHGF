import numpy as np
import numpy.linalg as la
from sklearn.decomposition import PCA
from scipy.linalg import cholesky
from numpy.linalg import LinAlgError
from numpy.random import standard_normal
from functions import *
import matplotlib.pyplot as plt


class ASEBO:
    kind = 'Adaptive ES-Active Subspaces'

    def __init__(self,
                 lr=1e-4,
                 sigma=1e-4,
                 k=50,
                 lambd=0.1,
                 thresh=10**(-4),
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
        self.lambd = lambd
        self.thresh = thresh
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

        alpha = 1
        G = []

        if debug:
            print('algorithm:', 'asebo', 'function:', function, 'dimension:',
                  len(x), 'initial value:', steps[0][1])

        for i in range(1, it + 1):
            try:
                if i % itprint == 0:
                    if debug:
                        print(i, 'th iteration - value:', steps[i - 1][1],
                              'last best value:', best_value)

                grad, alpha = self.grad_estimator(x, G, i, alpha, f)

                if i == 1:
                    G = np.array(grad)
                else:
                    G *= 0.99
                    G = np.vstack([G, grad])

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

    def grad_estimator(self, x, G, i, alpha, f):
        '''
        Performs the Central Gaussian Smoothing around x for the function f.

        Input:
        - x: point on which the operation is performed, array
        - G: buffer of gradients, 2-D array
        - i: iteration, int
        - alpha: momentum of covariance, float
        - f: function over which the smoothing is performed, function object

        Output:
        - the estimated gradient, array
        '''
        if i >= self.k:
            pca = PCA()
            pca_fit = pca.fit(G)
            var_exp = pca_fit.explained_variance_ratio_
            var_exp = np.cumsum(var_exp)
            n_samples = np.argmax(var_exp > self.thresh) + 1

            if n_samples < 10:
                n_samples = 10

            U = pca_fit.components_[:n_samples]
            UUT = np.matmul(U.T, U)
            U_ort = pca_fit.components_[n_samples:]
            UUT_ort = np.matmul(U_ort.T, U_ort)

            if i == self.k:
                n_samples = 100

        else:
            UUT = np.zeros([len(x), len(x)])
            alpha = 1
            n_samples = 100

        cov = (alpha / len(x)) * np.eye(len(x)) + (
            (1 - alpha) / n_samples) * UUT
        cov *= self.sigma
        A = np.zeros((n_samples, len(x)))

        try:
            l = cholesky(cov, check_finite=False, overwrite_a=True)
            for j in range(n_samples):
                try:
                    A[j] = np.zeros(len(x)) + l.dot(standard_normal(len(x)))

                except LinAlgError:
                    A[j] = np.random.randn(len(x))

        except LinAlgError:
            for j in range(n_samples):
                A[j] = np.random.randn(len(x))

        A /= np.linalg.norm(A, axis=-1)[:, np.newaxis]

        grad = np.zeros((len(x), ))
        evaluations = []
        for j in range(n_samples):
            dire = A[j].reshape(x.shape)
            dir_plus = x + self.sigma * dire
            dir_minus = x - self.sigma * dire

            evaluations_plus = f.evaluate(dir_plus)
            evaluations_minus = f.evaluate(dir_minus)

            evaluations.append(evaluations_plus)
            evaluations.append(evaluations_minus)

            grad = grad + (evaluations_plus -
                           evaluations_minus) * dire.reshape(grad.shape)

        grad /= (2 * self.sigma)

        if i >= self.k:
            alpha = np.linalg.norm(np.dot(grad, UUT_ort)) / np.linalg.norm(
                np.dot(grad, UUT))

        return grad, alpha

    def plot_results(self, values, function):
        '''
        Plots the given values
        '''
        plt.plot(values, 'r', label=ASEBO.kind)
        plt.yscale('log')
        plt.legend()
        plt.title(function)
        plt.show()
        pass