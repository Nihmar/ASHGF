"""Adapter driving the frozen original prototype without modifying it.

The modules in ``original_python_code/`` are loaded unmodified through
``sys.path``. A minimal stand-in for legacy ``gym`` is installed when the real
package is absent (benchmark functions never touch it). Objective calls are
counted by temporarily wrapping ``Function.evaluate`` on the shared original
class object.

Documented quirks of the prototype surfaced here:

* the returned value list always drops the final iterate's value (off-by-one);
  :func:`run_original` reconstructs the complete sequence from the call stream;
* the prototype swallows exceptions per iteration (prints "Something has gone
  wrong!" and stops); notably its SGES guidance phase calls
  ``int(np.random.choice(...))``, which raises under NumPy >= 2, so under modern
  environments the frozen SGES aborts at the first post-warm-up iteration.
* global ``np.random.seed`` per run (legacy MT19937 stream, unlike the new
  package's PCG64 generators) — trajectories are not bit-comparable for equal
  seeds unless ``x_init`` and the direction stream are aligned explicitly;
* no internal function-evaluation counter (added here);
* mutable class-level ``data`` dicts in ASGF/ASHGF shared across instances
  (restored after each run when overrides are applied).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ORIGINAL_ALGORITHMS = ("gd", "sges", "asebo", "asgf", "ashgf")

_CLASS_NAMES = {"gd": "GD", "sges": "SGES", "asebo": "ASEBO", "asgf": "ASGF", "ashgf": "ASHGF"}

#: Constructor defaults matching the thesis hyperparameters where applicable.
DEFAULT_CTOR_PARAMS: dict[str, dict[str, float | int]] = {
    "gd": {"lr": 1e-4, "sigma": 1e-4},
    "sges": {
        "lr": 1e-4,
        "sigma": 1e-4,
        "k": 50,
        "k1": 0.9,
        "k2": 0.1,
        "alpha": 0.5,
        "delta": 1.1,
        "t": 50,
    },
    "asebo": {"lr": 1e-4, "sigma": 1e-4, "k": 50, "lambd": 0.1, "thresh": 1e-4},
    "asgf": {},
    "ashgf": {"k1": 0.9, "k2": 0.1, "alpha": 0.5, "delta": 1.1, "t": 50},
}


@dataclass(frozen=True)
class OriginalRecord:
    """Structured record of one frozen-prototype run (comparable to RunResult)."""

    algorithm: str
    function: str
    dim: int
    seed: int
    n_iterations: int  # fully completed iterations
    values: tuple[float, ...]  # iterate sequence covered by ``values``
    best_value: float  # minimum over ``values`` (benchmarks minimize)
    total_function_evaluations: int
    wall_time_seconds: float
    terminated_normally: bool  # False if the prototype aborted on an internal error
    fes_per_iterate: tuple[int, ...] | None  # exact cumulative FEs aligned with values


def _original_dir() -> Path:
    env = os.environ.get("ASHGF_ORIGINAL_DIR")
    if env:
        return Path(env)
    # src/ashgf/parity/original_runner.py -> repo root is three parents up.
    return Path(__file__).resolve().parents[3] / "original_python_code"


@contextlib.contextmanager
def _gym_stub():
    """Make ``import gym`` succeed when legacy gym is not installed."""
    try:
        import gym  # noqa: F401

        yield
        return
    except ImportError:
        pass

    stub = types.ModuleType("gym")

    def _unavailable(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "legacy gym is not installed; RL environments cannot be used in parity runs"
        )

    stub.make = _unavailable
    previous = sys.modules.get("gym")
    sys.modules["gym"] = stub
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("gym", None)
        else:
            sys.modules["gym"] = previous


def _load_original_module(name: str, directory: Path):
    """Import an unmodified original module under a private cache key."""
    cache_key = f"_ashgf_orig_{name}"
    cached = sys.modules.get(cache_key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(cache_key, directory / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {directory / (name + '.py')}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[cache_key] = module
    saved_path = list(sys.path)
    sys.path.insert(0, str(directory))
    try:
        importlib.import_module("functions")  # shared by every original module
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
    return module


def run_original(
    algorithm: str,
    *,
    function: str = "sphere",
    dim: int = 10,
    it: int = 100,
    seed: int = 2003,
    x_init: NDArray[np.float64] | None = None,
    eps: float = 1e-8,
    ctor_params: dict[str, float | int] | None = None,
    data_overrides: dict[str, float | int] | None = None,
) -> OriginalRecord:
    """Run one frozen-prototype optimization and collect a structured record.

    Parameters
    ----------
    algorithm:
        One of :data:`ORIGINAL_ALGORITHMS` (module names of the prototype).
    function, dim, it, seed, x_init, eps:
        Forwarded to the prototype's ``optimize`` call.
    ctor_params:
        Overrides for constructor parameters (see :data:`DEFAULT_CTOR_PARAMS`).
    data_overrides:
        Temporary overrides of the mutable class-level ``data`` dict used by
        ASGF/ASHGF (restored afterwards).
    """
    if algorithm not in ORIGINAL_ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; expected one of {ORIGINAL_ALGORITHMS}")
    directory = _original_dir()
    with _gym_stub():
        mod = _load_original_module(algorithm, directory)
    cls = getattr(mod, _CLASS_NAMES[algorithm])

    params: dict[str, float | int] = {"seed": seed, "eps": eps}
    params.update(DEFAULT_CTOR_PARAMS[algorithm])
    if ctor_params:
        params.update(ctor_params)

    saved_data: dict[str, object] = {}
    if data_overrides and hasattr(cls, "data"):
        for key, value in data_overrides.items():
            saved_data[key] = cls.data.get(key)
            cls.data[key] = value

    counter = {"n": 0}
    stream: list[float] = []
    grad_counts: list[int] = []
    original_evaluate = mod.Function.evaluate

    def counting_evaluate(self, x) -> float:
        counter["n"] += 1
        value = original_evaluate(self, x)
        stream.append(float(value))
        return value

    mod.Function.evaluate = counting_evaluate
    t0 = time.perf_counter()
    buffer = io.StringIO()
    try:
        instance = cls(**params)
        # Every completed iteration calls grad_estimator exactly once and evaluates
        # the new point once afterwards, so snapshotting the evaluation count right
        # after each grad_estimator call yields the exact cumulative FE per iterate.
        original_grad = instance.grad_estimator

        def counting_grad(*args: object, **kwargs: object):
            result = original_grad(*args, **kwargs)
            grad_counts.append(counter["n"])
            return result

        instance.grad_estimator = counting_grad
        with contextlib.redirect_stdout(buffer):
            _, returned_values = instance.optimize(
                function=function, dim=dim, it=it, x_init=x_init, debug=False
            )
    finally:
        mod.Function.evaluate = original_evaluate
        for key, value in saved_data.items():
            cls.data[key] = value
    wall_time = time.perf_counter() - t0

    terminated_normally = "Something has gone wrong!" not in buffer.getvalue()
    values = [float(v) for v in returned_values]
    # Prototype quirk: its returned list always misses the final iterate's value;
    # on normal termination the last entry of the evaluation stream is exactly that
    # value. On a mid-iteration crash no new iterate was recorded, so nothing is
    # appended.
    if terminated_normally and len(stream) > len(values):
        values.append(stream[-1])
    best_value = min(values)

    fes: tuple[int, ...] | None = None
    if len(grad_counts) == len(values) - 1:
        candidate = [1] + [c + 1 for c in grad_counts]
        if candidate[-1] == counter["n"]:
            fes = tuple(candidate)
    return OriginalRecord(
        algorithm=algorithm,
        function=function,
        dim=dim,
        seed=seed,
        n_iterations=len(values) - 1,
        values=tuple(values),
        best_value=best_value,
        total_function_evaluations=counter["n"],
        wall_time_seconds=wall_time,
        terminated_normally=terminated_normally,
        fes_per_iterate=fes,
    )
