# Economics Context

This file provides domain knowledge needed to work with the models in this
repository. Agents working on this codebase should understand these concepts to
avoid introducing economically nonsensical changes.

## The life-cycle consumption-saving problem

A consumer lives from age 25 to (at most) 120, faces uncertain income, and must
decide how much to consume vs. save each period. The consumer retires at age 65.
The goal: maximize expected lifetime utility of consumption (and possibly
bequests), discounted by a time preference factor.

### Key economic parameters

- **CRRA (rho)**: Coefficient of Relative Risk Aversion. Higher values mean the
  consumer is more averse to consumption fluctuations. Typical estimated values
  are 2–10. Must be > 1 for the model to behave well.
- **DiscFac (beth)**: Discount factor adjustment. Multiplied by an age-varying
  sequence. Values near 1.0 mean the consumer values future consumption almost
  as much as current. Must be in (0.5, 1.1).
- **Rfree**: Risk-free interest rate (gross, so 1.03 = 3% return).

### Income process

Uses Cagetti (2003) calibration for college-educated households:
- Permanent income grows deterministically with age (hump-shaped)
- Subject to permanent and transitory shocks (log-normal)
- Unemployment risk: 5% probability with 30% replacement rate while working
- Post-retirement: much smaller shocks, 0.5% "unemployment" probability

### Wealth-to-income ratio

The key moment matched in estimation. For each age group, compute the median
ratio of financial wealth (`bNrm`, normalized by permanent income) across
simulated agents, and compare to the SCF empirical median.

## Model variants

### IndShock (baseline)

Standard buffer-stock saving model. Only parameters: CRRA (and possibly DiscFac).
Consumer chooses consumption; remainder is saved at the risk-free rate.

### Portfolio

Adds choice between a risky asset (stocks) and risk-free asset (bonds) each
period. Additional moments to match: equity portfolio share by age from S&P
target-date fund glidepath data.

The `post_solve` method sets `cFunc = cFuncAdj` (the consumption function
conditional on adjusting the portfolio) because the model assumes costless
portfolio rebalancing.

### WarmGlow (bequest motive)

Adds a "warm glow" bequest utility: the consumer gets utility from leaving
wealth at death, parameterized by:
- **BeqMPC**: Pseudo marginal propensity to consume from bequests. Internally
  transformed: `BeqFac = BeqMPC^(-CRRA)`
- **BeqInt**: Pseudo intercept. Internally transformed:
  `BeqShift = BeqInt / BeqMPC`

These reparameterizations ensure the optimizer searches over economically
meaningful quantities rather than the raw structural parameters.

### WarmGlowPortfolio

Combines bequest motive and portfolio choice. Estimates CRRA, BeqMPC, BeqInt,
and matches both wealth and portfolio share moments.

### WealthPortfolio

An alternative to bequests: utility is defined over both consumption AND wealth
holdings directly (wealth in the utility function). Parameters:
- **WealthShare**: Weight on wealth in utility (0 to 1)
- **WealthShift**: Shift parameter for wealth utility

## Subjective beliefs

Optional extensions add subjective (potentially biased) beliefs about:
- **Stock market**: Consumer may perceive different risk/return than reality.
  Model is solved with subjective beliefs but simulated with true parameters.
  Calibrated from Mateo Velasquez-Giraldo's JMP.
- **Labor market**: Consumer may perceive different income risk. Calibrated from
  Tao Wang's JMP.

Activated via the agent name string: `"Sub(Stock)"`, `"Sub(Labor)"`, or
`"Sub(Stock)(Labor)"`.

## Data sources

### SCF 2004 (`src/data/SCFdata.csv`)

Survey of Consumer Finances, 2004 wave. Contains:
- `wealth_income_ratio`: Financial wealth / permanent income
- `age_group`: Age bracket labels matching `age_mapping`
- `weight`: Survey sampling weights

Weighted medians are computed using `statsmodels.DescrStatsW`.

### S&P Target Date Glidepath (`src/data/S&P Target Date glidepath.xlsx`)

Equity allocation percentages by age from S&P target-date index funds. Used as
empirical portfolio share moments for Portfolio-type models.

### Cagetti 2003

Income process calibration (growth rates, shock variances by age). Accessed
through HARK's `Calibration.Income.IncomeTools.Cagetti_income`.

## Notation conventions

The paper and code use Econ-ARK notation extensively. Key symbols defined as
LaTeX macros in `myst.yml`:
- `\CRRA` (rho) — relative risk aversion
- `\DiscFac` (beta) — discount factor
- `\Rfree` — risk-free gross return
- `\PermGroFac` — permanent income growth factor
- `\MPC` — marginal propensity to consume
- `\wealth` — wealth level
