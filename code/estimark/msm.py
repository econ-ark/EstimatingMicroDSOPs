from pathlib import Path
from time import time

import estimagic as em

from estimark.estimation import (
    get_empirical_moments,
    get_initial_guess,
    get_moments_cov,
    make_agent,
    save_results,
    simulate_moments,
)

# Parameters for the consumer type and the estimation
from estimark.parameters import (
    init_params_options,
    minimize_options,
)

# SCF 2004 data on household wealth

# Pathnames to the other files:
# Relative directory for primitive parameter files
tables_dir = "content/tables/msm/"
Path(tables_dir).mkdir(parents=True, exist_ok=True)
# Relative directory for primitive parameter files
figures_dir = "content/figures/msm/"
Path(figures_dir).mkdir(parents=True, exist_ok=True)


def estimate_msm(
    init_agent_name,
    params_to_estimate=None,
    subjective_stock=False,
    subjective_labor=False,
):
    ############################################################
    # Make agent
    ############################################################

    agent = make_agent(
        init_agent_name=init_agent_name,
        subjective_stock=subjective_stock,
        subjective_labor=subjective_labor,
    )

    print("Agent created: ", agent.name)

    ############################################################
    # Get empirical moments
    ############################################################

    emp_moments = get_empirical_moments(agent.name)

    print("Calculated empirical moments.")

    ############################################################
    # Get moments covariance matrix
    ############################################################

    moments_cov = get_moments_cov(agent.name, emp_moments)

    print("Calculated moments covariance matrix.")

    ############################################################
    # Get initial guess
    ############################################################

    initial_guess = get_initial_guess(agent.name, params_to_estimate, tables_dir)

    print("Estimating MSM...")

    upper_bounds = {
        key: value
        for key, value in init_params_options["upper_bounds"].items()
        if key in initial_guess
    }

    lower_bounds = {
        key: value
        for key, value in init_params_options["lower_bounds"].items()
        if key in initial_guess
    }

    t0 = time()

    res = em.estimate_msm(
        simulate_moments,
        emp_moments,
        moments_cov,
        initial_guess,
        upper_bounds=upper_bounds,
        lower_bounds=lower_bounds,
        optimize_options=minimize_options,
        simulate_moments_kwargs={"agent": agent, "emp_moments": emp_moments},
    )

    time_to_estimate = time() - t0

    print("MSM estimation complete.")

    save_results(res, agent.name, time_to_estimate, tables_dir, "_params")


if __name__ == "__main__":
    # Set booleans to determine which tasks should be done
    # Which agent type to estimate ("IndShock" or "Portfolio")
    local_agent_name = "Portfolio"

    # Whether to use subjective beliefs
    local_subjective_stock = False
    local_subjective_labor = False

    local_params_to_estimate = ["CRRA", "DiscFac"]

    estimate_msm(
        init_agent_name=local_agent_name,
        params_to_estimate=local_params_to_estimate,
        subjective_stock=local_subjective_stock,
        subjective_labor=local_subjective_labor,
    )
