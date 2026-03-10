# AGENTS.md — EstimatingMicroDSOPs (estimark)

This repository implements structural estimation of life-cycle
consumption-saving models using the [HARK](https://github.com/econ-ark/HARK)
toolkit and [estimagic](https://estimagic.readthedocs.io/) optimizer. It matches
simulated wealth-to-income ratio profiles against median holdings from the 2004
Survey of Consumer Finances (SCF), organized around five agent types of
increasing complexity.

See `.agents/` for detailed guides on specific subsystems.

## Quick orientation

| Path               | Purpose                                                      |
| ------------------ | ------------------------------------------------------------ |
| `src/estimark/`    | Installable Python package (estimation, agents, calibration) |
| `src/do_all.py`    | Interactive CLI entry point for running replications         |
| `src/notebooks/`   | Jupyter notebooks for individual model estimations           |
| `content/paper/`   | Research paper in MyST Markdown (`01-paper.md`)              |
| `content/tables/`  | Generated CSV estimation results (min/, msm/, TRP/)          |
| `content/figures/` | Generated SVG/PNG/PDF figures                                |
| `tests/`           | pytest tests (minimal — just version check currently)        |
| `myst.yml`         | MyST site config with extensive LaTeX math macros            |
| `noxfile.py`       | Nox sessions: lint, pylint, tests, docs, build               |

## Key conventions

- **Package name** is `estimark`; repo name is `EstimatingMicroDSOPs`.
- **Build system**: Hatchling + hatch-vcs. Version derived from git tags,
  written to `src/estimark/_version.py`.
- **No core dependencies** are declared in `[project.dependencies]`. Runtime
  deps live in `[project.optional-dependencies].run` (numpy, pandas, matplotlib,
  scipy, statsmodels, estimagic, HARK from GitHub main).
- **Code style**: ruff for linting/formatting, mypy for type checking, codespell
  for spell checking. All enforced via pre-commit (`.pre-commit-config.yaml`).
- **Pre-commit excludes** CSV files under `content/tables/` since they are
  generated output.
- **Python version support**: 3.8–3.13 (per pyproject.toml classifiers and CI
  matrix).
- Imports use `from __future__ import annotations` throughout.

## Agent types (model hierarchy)

1. **IndShock** — Standard income-shock life-cycle model (CRRA only)
2. **Portfolio** — Adds portfolio choice between risky and risk-free assets
3. **WarmGlow** — Adds warm-glow bequest motive (BeqMPC, BeqInt params)
4. **WarmGlowPortfolio** — Bequest motive + portfolio choice
5. **WealthPortfolio** — Wealth-in-utility + portfolio choice (WealthShare,
   WealthShift params)

Each agent class in `agents.py` inherits from a `TempConsumerType` mixin (custom
`sim_birth`/`sim_death`) and the corresponding HARK consumer type via multiple
inheritance.

## Parameters estimated

| Parameter     | Description                           | Typical bounds |
| ------------- | ------------------------------------- | -------------- |
| `CRRA`        | Coefficient of relative risk aversion | [1.1, 20.0]    |
| `DiscFac`     | Discount factor adjustment            | [0.5, 1.1]     |
| `BeqMPC`      | Bequest pseudo-MPC                    | [0.0, 1.0]     |
| `BeqInt`      | Bequest pseudo-intercept              | [0.0, 10.0]    |
| `WealthShare` | Wealth-in-utility share               | [0.01, 0.99]   |
| `WealthShift` | Wealth-in-utility shift               | [0.0, 100.0]   |

## Running estimations

```bash
# Interactive
python src/do_all.py

# Direct (example: WarmGlowPortfolio with min-distance)
python src/estimark/estimation.py

# Batch TRP models
python src/run_all.py

# Batch MSM with Dask parallelism
python src/run_all_msm.py
```

## Common tasks for agents

- **Adding a new agent type**: Subclass `TempConsumerType` + the HARK type in
  `agents.py`, add to `agent_types` dict in `estimation.py`, update
  `params_to_estimate` in `options.py` if new parameters are introduced.
- **Changing calibration**: Edit `parameters.py`. Key dicts are
  `init_calibration`, `init_params_options`, `age_mapping`, `sim_mapping`.
- **Modifying the paper**: Edit `content/paper/01-paper.md` (MyST Markdown).
  Math macros are defined in `myst.yml`. Build with `myst build`.
- **Running CI locally**: `nox -s lint`, `nox -s tests`, or
  `pre-commit run --all-files`.

## Files agents should not modify without care

- `content/tables/` — Generated output; regenerate by running estimations
- `src/data/` — Empirical data files (SCF, S&P glidepath, Cagetti)
- `myst.yml` math macros — Shared notation; changes affect entire paper
- `.copier-answers.yml` — Template metadata from scientific-python/cookie
