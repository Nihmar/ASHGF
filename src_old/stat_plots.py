import numpy as np
from gd import GD
from sges import SGES
from asebo import ASEBO
from asgf import ASGF
from ashgf import ASHGF
import matplotlib.pyplot as plt
import sys
import os
import pickle

np.random.seed(0)

it = 10000
debug_it = 100
debug = True
dim = 100
x_0 = np.random.randn(dim)

functions = ['sphere', 'levy', 'rastrigin', 'ackley']

algorithms = {
    'GD': GD,
    'SGES': SGES,
    'ASGF': ASGF,
    'ASHGF': ASHGF,
    'ASEBO': ASEBO
}

for algorithm in algorithms:
    for function in functions:
        bests = []

        if 'descents_{}.pkl'.format(function) not in os.listdir(
                os.path.join('code', 'results', 'stats', algorithm)):

            for i in range(100):
                print(i)
                r_seed = np.random.randint(0, 10000)
                alg = algorithms[algorithm](seed=r_seed)
                alg_best, alg_all = alg.optimize(function, dim, it, x_0, debug,
                                                 debug_it)
                bests.append(alg_all)

            output = open(
                os.path.join('code', 'results', 'stats', algorithm,
                             'descents_{}.pkl'.format(function)), 'wb')
            pickle.dump(bests, output)
            output.close()

        print('Finished {} for algorithm {}'.format(function, algorithm))


for algorithm in algorithms:
    for function in functions:

        if '{}_convergence.png'.format(function) not in os.listdir(
                os.path.join('code', 'results', 'stats', algorithm)):

            pkl_file = open(
                os.path.join('code', 'results', 'stats', algorithm,
                            'descents_{}.pkl'.format(function)), 'rb')
            bests = pickle.load(pkl_file)
            pkl_file.close()

            for i, best in enumerate(bests):
                plt.plot(best, label=algorithm + '_' + str(i))

            plt.yscale('log')
            plt.title(function)
            plt.xlabel(r'Iterations $t$')
            plt.ylabel(r'$f(x_t)$')
            plt.savefig(os.path.join('code', 'results', 'stats', algorithm,
                                    '{}_convergence.png'.format(function)),
                        dpi=600)
            plt.show()

            min_descent = []
            max_descent = []
            mean_descent = []
            std_plus_descent = []
            std_minus_descent = []

            min_num_it = int(np.max([len(best) for best in bests]))

            for i in range(min_num_it):
                values = [best[i] if i < len(best) else np.nan for best in bests]
                mean_value = np.nanmean(values)
                min_value = np.nanmin(values)
                max_value = np.nanmax(values)
                std_value = np.nanstd(values)

                min_descent.append(min_value)
                max_descent.append(max_value)
                mean_descent.append(mean_value)
                std_plus_descent.append(mean_value + std_value)
                std_minus_descent.append(max([min_value, mean_value - std_value]))

            plt.figure()
            plt.plot(min_descent, label='min')
            plt.plot(max_descent, label='max')
            plt.plot(mean_descent, label='mean')
            plt.fill_between(range(min_num_it),
                            std_minus_descent,
                            std_plus_descent,
                            alpha=0.5)
            plt.yscale('log')
            plt.legend()
            plt.title(function)
            plt.xlabel(r'Iterations $t$')
            plt.ylabel(r'$f(x_t)$')
            plt.savefig(os.path.join('code', 'results', 'stats', algorithm,
                                    '{}_convergence_mean.png'.format(function)),
                        dpi=600)
            plt.show()

        print('Saved images for {}'.format(function))
