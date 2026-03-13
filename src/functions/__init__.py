from typing import Callable, Dict

import numpy as np

from .benchmarks import *


class Function:
    MAX_VALUE = 1e30
    CLIP_RANGE = 1e10

    def __init__(self, name: str = "sphere"):
        """
        Initialize the Function wrapper.

        Args:
            name: Name of the function to optimize.
        """
        self.name = name

    functions_list: Dict[str, Callable[['Function', np.ndarray], float]] = {
        "extended_feudenstein_and_roth": extended_feudenstein_and_roth,
        "extended_trigonometric": extended_trigonometric,
        "extended_rosenbrock": extended_rosenbrock,
        "generalized_rosenbrock": generalized_rosenbrock,
        "extended_white_and_holst": extended_white_and_holst,
        "extended_baele": extended_baele,
        "extended_penalty": extended_penalty,
        "perturbed_quadratic": perturbed_quadratic,
        "raydan_1": raydan_1,
        "raydan_2": raydan_2,
        "diagonal_1": diagonal_1,
        "diagonal_2": diagonal_2,
        "diagonal_3": diagonal_3,
        "hager": hager,
        "extended_tridiagonal_1": extended_tridiagonal_1,
        "diagonal_4": diagonal_4,
        "diagonal_5": diagonal_5,
        "extended_himmelblau": extended_himmelblau,
        "generalized_white_and_holst": generalized_white_and_holst,
        "extended_psc1": extended_psc1,
        "extended_bd1": extended_bd1,
        "extended_maratos": extended_maratos,
        "extended_cliff": extended_cliff,
        "perturbed_quadratic_diagonal": perturbed_quadratic_diagonal,
        "extended_hiebert": extended_hiebert,
        "quadratic_qf1": quadratic_qf1,
        "extended_quadratic_penalty_qp1": extended_quadratic_penalty_qp1,
        "extended_quadratic_penalty_qp2": extended_quadratic_penalty_qp2,
        "quadratic_qf2": quadratic_qf2,
        "extended_quadratic_exponential_ep1": extended_quadratic_exponential_ep1,
        "extended_tridiagonal_2": extended_tridiagonal_2,
        "fletcbv3": fletcbv3,
        "fletchcr": fletchcr,
        "bdqrtic": bdqrtic,
        "tridia": tridia,
        "arwhead": arwhead,
        "nondia": nondia,
        "nondquar": nondquar,
        "dqdrtic": dqdrtic,
        "eg2": eg2,
        "broyden_tridiagonal": broyden_tridiagonal,
        "almost_perturbed_quadratic": almost_perturbed_quadratic,
        "liarwhd": liarwhd,
        "power": power,
        "engval1": engval1,
        "edensch": edensch,
        "indef": indef,
        "cube": cube,
        "bdexp": bdexp,
        "genhumps": genhumps,
        "mccormck": mccormck,
        "nonscomp": nonscomp,
        "vardim": vardim,
        "quartc": quartc,
        "sinquad": sinquad,
        "extended_denschnb": extended_denschnb,
        "extended_denschnf": extended_denschnf,
        "dixon3dq": dixon3dq,
        "cosine": cosine,
        "sine": sine,
        "biggsb1": biggsb1,
        "generalized_quartic": generalized_quartic,
        "diagonal_7": diagonal_7,
        "diagonal_8": diagonal_8,
        "fh3": fh3,
        "sincos": sincos,
        "diagonal_9": diagonal_9,
        "himmelbg": himmelbg,
        "himmelh": himmelh,
        "ackley": ackley,
        "griewank": griewank,
        "levy": levy,
        "rastrigin": rastrigin,
        "schwefel": schwefel,
        "sphere": sphere,
        "sum_of_different_powers": sum_of_different_powers,
        "trid": trid,
        "zakharov": zakharov
    }

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function at point x.

        Args:
            x: Point at which to evaluate the function.

        Returns:
            Function value at x.
        """
        x_clipped = np.clip(x, -self.CLIP_RANGE, self.CLIP_RANGE)
        
        func = self.functions_list[self.name]
        result = func(x_clipped)
        
        if not np.isfinite(result):
            return self.MAX_VALUE
        
        return np.clip(result, -self.MAX_VALUE, self.MAX_VALUE)

    def __call__(self, x: np.ndarray) -> float:
        """Make the Function object callable."""
        return self.evaluate(x)
