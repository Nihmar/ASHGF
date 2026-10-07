from functions import *
from gd import GD
from sges import SGES
from asebo import ASEBO
from asgf import ASGF
from ashgf import ASHGF
import os
from os import path
import pandas as pd

main_folder = path.join('code', 'results', 'profiles')
debug_it = 100
debug = False
tau = 10**(-3)
it = 10000
dims = [10, 100, 1000]
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


def create_folders():
    functions = get_functions()

    for dim in dims:
        if not str(dim) in os.listdir(main_folder):
            os.mkdir(path.join(main_folder, str(dim)))

        for function in functions:
            if not function in os.listdir(path.join(main_folder, str(dim))):
                os.mkdir(path.join(main_folder, str(dim), function))

            for algorithm in algorithms.keys():
                if not algorithm in os.listdir(
                        path.join(main_folder, str(dim), function)):
                    os.mkdir(
                        path.join(main_folder, str(dim), function, algorithm))


def evaluations(it, dim, name):
    correction = 1

    if name in ['ASGF', 'ASHGF']:
        return it + it * dim * 4
    else:
        return (it + it * dim * 2) * correction


def iterations(evs, dim, name):
    if name in ['ASGF', 'ASHGF']:
        return int(evs / (1 + dim * 4))
    else:
        return int(evs / (1 + dim * 2))


def execute_algorithm(function, dim, name, algorithm):
    if name not in ['ASGF', 'ASHGF']:
        alg = algorithm(1e-4, 1e-4)
    else:
        alg = algorithm()
    alg_best, alg_all = alg.optimize(function, dim, it, None,
                                                      debug, debug_it)

    return alg_best, alg_all


def save_data(function, dim, name, algorithm):
    alg_best, alg_all = execute_algorithm(
        function, dim, name, algorithm)

    Series = pd.Series(alg_all, name='descent')
    Series.to_csv(path.join('code', 'results', str(dim), function, name,
                            'descent.csv'),
                  header=True,
                  index=False)


def main():
    fs = get_functions()
    create_folders()

    for dim in dims:
        for name, algorithm in algorithms.items():
            for function in fs:
                if not 'descent.csv' in os.listdir(
                        path.join('code', 'results', 'profiles', str(dim), function,
                                  name)):
                    save_data(function, dim, name, algorithm)

                print('Done with: {} - {} - {}'.format(name, function, dim))


main()