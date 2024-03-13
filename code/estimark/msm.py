import csv
from pathlib import Path
from time import time

import estimagic as em
import numpy as np

from estimark.estimation import (
    get_empirical_moments,
    get_initial_guess,
    get_weighted_moments,
    make_agent,
    simulate_moments,
    save_results,
)

# Parameters for the consumer type and the estimation
from estimark.parameters import (
    age_mapping,
    init_params_options,
    minimize_options,
)

# SCF 2004 data on household wealth
from estimark.scf import scf_data

msm_dir = "content/tables/msm/"
Path(msm_dir).mkdir(parents=True, exist_ok=True)


def get_moments_cov(agent_name, emp_moments):
    moments_cov = em.get_moments_cov(
        scf_data,
        get_weighted_moments,
        moment_kwargs={
            "variable": "wealth_income_ratio",
            "weights": "weight",
            "groups": "age_group",
            "mapping": age_mapping,
        },
    )

    if "Port" in agent_name:
        # how many keys in emp_moments contain "_port"
        n_port = sum("_port" in key for key in emp_moments.keys())
        share_moments_cov = np.diag(np.ones(n_port))

        moments_cov = np.block(
            [
                [moments_cov, np.zeros((moments_cov.shape[0], n_port))],
                [np.zeros((n_port, moments_cov.shape[1])), share_moments_cov],
            ],
        )

    return moments_cov


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

    initial_guess = get_initial_guess(agent.name, params_to_estimate, msm_dir)

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

    save_results(res, agent.name, time_to_estimate, msm_dir, "_params")


if __name__ == "__main__":
    # Set booleans to determine which tasks should be done
    # Which agent type to estimate ("IndShock" or "Portfolio")
    local_agent_name = "IndShock"

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
