import numpy as np
from functions import *
from gd import GD
from sges import SGES
from asebo import ASEBO
from asgf import ASGF
from ashgf import ASHGF
import matplotlib.pyplot as plt
import sys
# np.random.seed(69)

function = 'levy'
it = 5000
debug_it = 1000
dim = 10

gd = GD(0.5, 1)
gd_best, gd_all = gd.optimize(function, dim, it, None, True, debug_it)

ashgf = ASGF()
ashgf_best, ashgf_all = ashgf.optimize(function, dim, it, None, True, debug_it)

#------- Plot -------#
plt.plot([best[1] for best in gd_best], 'r', label='gd')
plt.plot([best[1] for best in ashgf_best], 'c', label='ashgf')

plt.yscale('log')
plt.legend()
plt.title(function)
plt.show()