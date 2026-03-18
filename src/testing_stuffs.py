import matplotlib.pyplot as plt  # noqa: F401
import warnings  # noqa: E402

# Ignore the specific deprecation warning from multiprocessing.forkserver
warnings.filterwarnings("ignore", category=DeprecationWarning, module="multiprocessing")

from optimizers.gd import GD  # noqa: E402, F401
from optimizers.sges import SGES  # noqa: E402, F401
from optimizers.asebo import ASEBO  # noqa: E402, F401
from optimizers.asgf import ASGF  # noqa: E402, F401
from optimizers.ashgf import ASHGF  # noqa: E402, F401

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
