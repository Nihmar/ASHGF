# ASHGF: Adaptive Stochastic Historical Gradient-Free Optimization

Repository for the master's thesis [*Adaptive Stochastic Historical Gradient-Free
Optimization*](https://thesis.unipd.it/handle/20.500.12608/21569) (Unipd).

It contains:

- `thesis/` — LaTeX source of the thesis (XeLaTeX, `Dissertate` class); entry point
  `thesis/dissertation.tex`.
- `original_python_code/` — the original research prototype (**frozen**, used only as
  behavioral ground truth).
- `src/ashgf/` — new production-quality Python implementation (under development):
  mathematically faithful to the thesis, vectorized, with exact function-evaluation counting.
- `tests/` — pytest suite (mathematical correctness vs analytic results, parity vs the
  frozen original).
- `comparisons/` — markdown reports and figures comparing old vs new implementations.
- `references/papers/` — PDFs of the SOTA papers (ASEBO, SGES, ASGF).

See [AGENTS.md](./AGENTS.md) for ground rules and conventions, and [PLAN.md](./PLAN.md)
for the project roadmap.

## Quick start

```bash
uv sync                  # install dependencies into .venv
uv run pytest            # run the test suite
uv run ruff check src tests
```
