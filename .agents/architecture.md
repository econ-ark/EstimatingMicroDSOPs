# Code Architecture

## Package layout (`src/estimark/`)

```
src/estimark/
├── __init__.py        # Exports __version__ (from hatch-vcs)
├── estimation.py      # Core estimation loop and all estimation functions
├── agents.py          # Life-cycle HARK consumer types (5 agent classes)
├── parameters.py      # All calibration values, bounds, age mappings
├── options.py         # Resource-level presets (low/medium/high/all)
├── scf.py             # Loads SCF 2004 wealth-income data from CSV
└── snp.py             # Loads S&P target-date glidepath from Excel
```

## Module dependency graph

```
do_all.py / run_all.py
    └── estimation.estimate()
            ├── agents.py        (make_agent)
            ├── parameters.py    (calibration, bounds, mappings)
            ├── scf.py           (empirical wealth data)
            ├── snp.py           (portfolio share targets)
            └── estimagic        (optimizer: em.minimize / em.estimate_msm)
                └── HARK         (agent.solve(), agent.simulate())
```

## Key design patterns

### Agent construction via multiple inheritance

Each agent type combines two base classes:
1. `TempConsumerType` — Custom `sim_birth` (draws from discrete initial wealth
   distribution) and `sim_death` (no death during simulation).
2. A HARK consumer type — Provides `solve()`, `simulate()`, consumption/portfolio
   functions.

```python
class PortfolioLifeCycleConsumerType(TempConsumerType, PortfolioConsumerType):
    ...
```

Python MRO means `TempConsumerType.sim_birth` overrides the HARK default.

### Agent name string as feature flag

The `agent.name` string (e.g. `"WarmGlowPortfolioSub(Stock)(Labor)Market"`)
drives conditional logic throughout `estimation.py`:
- `"Portfolio" in agent_name` → add share moments, track `Share` variable
- `"WarmGlow" in agent_name` → compute `BeqFac`/`BeqShift` from `BeqMPC`/`BeqInt`
- `"Sub" in agent_name` → use subjective belief calibrations
- `"(Stock)" in agent_name` → subjective stock market beliefs
- `"(Labor)" in agent_name` → subjective labor market beliefs

### Estimation result persistence

Results are saved as headerless two-column CSVs (`param_name, value`) in
`content/tables/{min,msm,TRP}/`. On subsequent runs, `get_initial_guess()`
reads these files to warm-start the optimizer.

## Scripts outside the package

| Script | Purpose |
|--------|---------|
| `src/do_all.py` | Interactive CLI: choose model, resource level, subjective beliefs |
| `src/run_all.py` | Batch run of TRP models (Portfolio, WealthPortfolio, WarmGlowPortfolio) |
| `src/run_all_msm.py` | MSM estimation for all agent types with Dask parallelism |
| `src/tests.py` | Legacy test script (broken import: `from estimark.min`; should be `estimark.estimation`) |

## Data files (`src/data/`)

| File | Contents |
|------|----------|
| `SCFdata.csv` | 2004 Survey of Consumer Finances: age, wealth-income ratio, weights |
| `S&P Target Date glidepath.xlsx` | S&P target-date fund equity share by age |
| `Cagetti2003.csv` | Cagetti (2003) income process calibration data |

## Content output (`content/`)

Estimation runs produce:
- `content/tables/min/` — Minimum-distance estimation CSVs
- `content/tables/msm/` — Method-of-simulated-moments CSVs
- `content/tables/TRP/` — Target Retirement Portfolio CSVs
- `content/figures/` — Sensitivity bar charts, SMM contour plots (SVG/PNG/PDF)

These are checked into version control as reproducibility artifacts.
