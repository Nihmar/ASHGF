# from functions import *
from gd import GD
from sges import SGES
from asebo import ASEBO
from asgf import ASGF
from ashgf import ASHGF
import os
from os import path
import pandas as pd
import pickle

debug = True
it = 10000
debug_it = 1000
seeds = [6177, 2832, 7361, 2778, 5416, 9652, 125, 978, 1487, 4156]
functions = {'RLenvironmentPendulum': 20, 'RLenvironmentCartPole': 20}
algorithms = {
    'GD': GD,
    'SGES': SGES,
    'ASGF': ASGF,
    'ASHGF': ASHGF,
    'ASEBO': ASEBO
}


def get_functions():
    lines = []
    with open(path.join('code', 'functions.txt')) as f:
        lines = f.readlines()

    fs = []
    for line in lines:
        if '*' in line:
            fs.append(line.replace('*', '')[:-1])

    return fs


def execute_algorithm(function, dim, name, algorithm, seed):
    if name not in ['ASGF', 'ASHGF']:
        alg = algorithm(1e-4, 1e-4, seed_env=seed)
    else:
        alg = algorithm(seed_env=seed)
    alg_best, alg_all = alg.optimize(function, dim, it, None, debug, debug_it)

    return alg_best, alg_all


def save_data(function, dim, name, algorithm, seed):
    alg_best, alg_all = execute_algorithm(function, dim, name, algorithm, seed)

    Series = pd.Series(alg_all, name='descent')
    Series.to_csv(path.join('code', 'results', 'RL', str(seed), function, name,
                            'descent.csv'),
                  header=True,
                  index=False)
    output = open(
        path.join('code', 'results', 'RL', str(seed), function, name,
                  'last_ev_{}_{}_{}.pkl'.format(seed, function, name)), 'wb')
    pickle.dump(alg_best, output)
    output.close()


def main():
    for function in get_functions():
        for algorithm in algorithms.keys():
            for seed in seeds:
                print('Begun with: {} - {} - {}'.format(algorithm, function,
                                                        functions[function]))
                if 'descent.csv' not in os.listdir(
                        path.join('code', 'results', 'RL', str(seed), function,
                                  algorithm)):
                    save_data(function, functions[function], algorithm,
                              algorithms[algorithm], seed)
                print('Done with: {} - {} - {}'.format(algorithm, function,
                                                    functions[function]))


main()