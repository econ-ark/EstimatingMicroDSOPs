# Estimation Pipeline

## Overview

The core loop: given parameter values, solve a life-cycle model, simulate many
agents, compute age-group median wealth-to-income ratios, and compare against
SCF data. The optimizer (estimagic) searches over parameters to minimize the
distance.

## Two estimation methods

### 1. Minimum-distance (`estimate_method="min"`)

Uses `em.minimize()` with a custom criterion function (`msm_criterion`) that
returns a scalar loss plus root-contributions for least-squares optimizers.

```
em.minimize(msm_criterion, initial_params, criterion_kwargs={agent, emp_moments, weights})
```

Default optimizer: `tranquilo_ls` with multistart enabled.

### 2. Method of Simulated Moments (`estimate_method="msm"`)

Uses `em.estimate_msm()` which handles the weighting matrix construction
internally from a moments covariance matrix.

```
em.estimate_msm(simulate_moments, emp_moments, moments_cov, initial_params)
```

## Step-by-step flow

### `estimation.estimate()` — the main entry point

1. **Create agent**: `make_agent(agent_name)` instantiates the appropriate HARK
   type with calibrated parameters from `parameters.init_calibration`.

2. **Get initial guess**: `get_initial_guess()` reads from a previous results
   CSV if available, otherwise falls back to `init_params_options["init_guess"]`.

3. **Get empirical moments**: `get_empirical_moments(agent_name)` computes
   weighted medians of wealth-income ratio by age group from SCF data. For
   Portfolio agents, also adds share moments from S&P glidepath data.

4. **Compute weights**: `calculate_weights()` normalizes by max wealth stat.
   Portfolio share moments get a separate weighting factor (currently set to 1.0).

5. **Run optimizer**: `do_estimate_model()` calls either `estimate_min()` or
   `estimate_msm()`. Results saved to CSV.

6. **Optional post-estimation**:
   - Bootstrap standard errors (`do_compute_se_boostrap`)
   - Sensitivity analysis via Jacobian (`do_compute_sensitivity`)
   - Contour plot of objective function (`do_make_contour_plot`)

### `simulate_moments(params, agent, emp_moments)` — the inner loop

Called many times by the optimizer. Each call:

1. `agent.assign_parameters(**params)` — update CRRA, DiscFac, etc.
2. Handle derived parameters (e.g. `BeqFac = BeqMPC^(-CRRA)`)
3. Handle subjective beliefs if applicable (swap in subjective distributions
   for solving, true distributions for simulating)
4. `agent.update()` → `agent.solve()` → `agent.initialize_sim()` → `agent.simulate()`
5. Extract `bNrm` (bank balances) history, compute median by age group
6. For Portfolio agents, also extract `Share` history
7. Return dict of simulated moments keyed by age-group labels

### `msm_criterion(params, agent, emp_moments, weights)` — the loss function

Computes weighted squared errors between simulated and empirical moments:
```
loss = sum( (weight_k * (sim_k - emp_k))^2  for k in moments )
```

Returns a dict with `value` (scalar loss), `contributions` (squared errors),
and `root_contributions` (signed errors) for least-squares optimizers.

## Age groups and mappings

Defined in `parameters.py`:
- **`age_mapping`**: Maps labels like `"(25,30]"` to arrays of real ages
  `[26, 27, 28, 29, 30]`
- **`sim_mapping`**: Same labels mapped to simulation time indices (age minus
  `initial_age=25`)
- Ages 61–70 are excluded from SCF matching (retirement transition)
- S&P share data only used for ages 71+

## Optimizer configuration (`parameters.minimize_options`)

```python
{
    "algorithm": "tranquilo_ls",
    "multistart": True,
    "algo_options": {
        "convergence.absolute_params_tolerance": 1e-6,
        "convergence.absolute_criterion_tolerance": 1e-6,
        "stopping.max_iterations": 100,
        "stopping.max_criterion_evaluations": 200,
        "n_cores": 12,
    },
}
```

The `n_cores: 12` setting is hardcoded — may need adjustment for different
machines.
