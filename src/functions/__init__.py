from typing import Callable, Dict

import numpy as np

import functions.benchmarks as _benchmarks


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

    functions_list: Dict[str, Callable[[np.ndarray], float]] = {
        "extended_feudenstein_and_roth": _benchmarks.extended_feudenstein_and_roth,
        "extended_trigonometric": _benchmarks.extended_trigonometric,
        "extended_rosenbrock": _benchmarks.extended_rosenbrock,
        "generalized_rosenbrock": _benchmarks.generalized_rosenbrock,
        "extended_white_and_holst": _benchmarks.extended_white_and_holst,
        "extended_baele": _benchmarks.extended_baele,
        "extended_penalty": _benchmarks.extended_penalty,
        "perturbed_quadratic": _benchmarks.perturbed_quadratic,
        "raydan_1": _benchmarks.raydan_1,
        "raydan_2": _benchmarks.raydan_2,
        "diagonal_1": _benchmarks.diagonal_1,
        "diagonal_2": _benchmarks.diagonal_2,
        "diagonal_3": _benchmarks.diagonal_3,
        "hager": _benchmarks.hager,
        "extended_tridiagonal_1": _benchmarks.extended_tridiagonal_1,
        "diagonal_4": _benchmarks.diagonal_4,
        "diagonal_5": _benchmarks.diagonal_5,
        "extended_himmelblau": _benchmarks.extended_himmelblau,
        "generalized_white_and_holst": _benchmarks.generalized_white_and_holst,
        "extended_psc1": _benchmarks.extended_psc1,
        "extended_bd1": _benchmarks.extended_bd1,
        "extended_maratos": _benchmarks.extended_maratos,
        "extended_cliff": _benchmarks.extended_cliff,
        "perturbed_quadratic_diagonal": _benchmarks.perturbed_quadratic_diagonal,
        "extended_hiebert": _benchmarks.extended_hiebert,
        "quadratic_qf1": _benchmarks.quadratic_qf1,
        "extended_quadratic_penalty_qp1": _benchmarks.extended_quadratic_penalty_qp1,
        "extended_quadratic_penalty_qp2": _benchmarks.extended_quadratic_penalty_qp2,
        "quadratic_qf2": _benchmarks.quadratic_qf2,
        "extended_quadratic_exponential_ep1": _benchmarks.extended_quadratic_exponential_ep1,
        "extended_tridiagonal_2": _benchmarks.extended_tridiagonal_2,
        "fletcbv3": _benchmarks.fletcbv3,
        "fletchcr": _benchmarks.fletchcr,
        "bdqrtic": _benchmarks.bdqrtic,
        "tridia": _benchmarks.tridia,
        "arwhead": _benchmarks.arwhead,
        "nondia": _benchmarks.nondia,
        "nondquar": _benchmarks.nondquar,
        "dqdrtic": _benchmarks.dqdrtic,
        "eg2": _benchmarks.eg2,
        "broyden_tridiagonal": _benchmarks.broyden_tridiagonal,
        "almost_perturbed_quadratic": _benchmarks.almost_perturbed_quadratic,
        "liarwhd": _benchmarks.liarwhd,
        "power": _benchmarks.power,
        "engval1": _benchmarks.engval1,
        "edensch": _benchmarks.edensch,
        "indef": _benchmarks.indef,
        "cube": _benchmarks.cube,
        "bdexp": _benchmarks.bdexp,
        "genhumps": _benchmarks.genhumps,
        "mccormck": _benchmarks.mccormck,
        "nonscomp": _benchmarks.nonscomp,
        "vardim": _benchmarks.vardim,
        "quartc": _benchmarks.quartc,
        "sinquad": _benchmarks.sinquad,
        "extended_denschnb": _benchmarks.extended_denschnb,
        "extended_denschnf": _benchmarks.extended_denschnf,
        "dixon3dq": _benchmarks.dixon3dq,
        "cosine": _benchmarks.cosine,
        "sine": _benchmarks.sine,
        "biggsb1": _benchmarks.biggsb1,
        "generalized_quartic": _benchmarks.generalized_quartic,
        "diagonal_7": _benchmarks.diagonal_7,
        "diagonal_8": _benchmarks.diagonal_8,
        "fh3": _benchmarks.fh3,
        "sincos": _benchmarks.sincos,
        "diagonal_9": _benchmarks.diagonal_9,
        "himmelbg": _benchmarks.himmelbg,
        "himmelh": _benchmarks.himmelh,
        "ackley": _benchmarks.ackley,
        "griewank": _benchmarks.griewank,
        "levy": _benchmarks.levy,
        "rastrigin": _benchmarks.rastrigin,
        "schwefel": _benchmarks.schwefel,
        "sphere": _benchmarks.sphere,
        "sum_of_different_powers": _benchmarks.sum_of_different_powers,
        "trid": _benchmarks.trid,
        "zakharov": _benchmarks.zakharov
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
