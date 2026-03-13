import numpy as np
import matplotlib.pyplot as plt

from optimizers.gd import GD
from optimizers.sges import SGES
from optimizers.asebo import ASEBO
from optimizers.asgf import ASGF
from optimizers.ashgf import ASHGF

# np.random.seed(69)

function = "levy"
it = 5000
debug_it = 1000
dim = 10

gd = GD(0.5, 1)
gd_best, gd_all = gd.optimize(function, dim, it, None, True, debug_it)

ashgf = ASGF()
ashgf_best, ashgf_all = ashgf.optimize(function, dim, it, None, True, debug_it)

# ------- Plot ------- #
plt.plot([best[1] for best in gd_best], "r", label="gd")
plt.plot([best[1] for best in ashgf_best], "c", label="ashgf")

plt.yscale("log")
plt.legend()
plt.title(function)
plt.show()