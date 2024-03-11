from pathlib import Path

import estimagic as em
import numpy as np
import pandas as pd

from estimark.estimation import (
    get_empirical_moments,
    make_agent,
)

# Parameters for the consumer type and the estimation
from estimark.parameters import (
    age_mapping,
    init_subjective_labor,
    init_subjective_stock,
    minimize_options,
    sim_mapping,
)

# SCF 2004 data on household wealth
from estimark.scf import scf_data
from estimark.snp import snp_data

msm_dir = "content/tables/msm"
Path(msm_dir).mkdir(parents=True, exist_ok=True)


def get_initial_guess(agent_name, init_guess):

    csv_path = Path(msm_dir) / f"{agent_name}_msm_results.csv"

    if csv_path.exists():
        initial_guess = pd.read_csv(csv_path)
        initial_guess = initial_guess.to_dict(orient="records")[0]
    else:
        initial_guess = init_guess

    return initial_guess


def simulate_moments(params, agent=None, emp_moments=None):
    """A quick check to make sure that the parameter values are within bounds.
    Far flung falues of DiscFac or CRRA might cause an error during solution or
    simulation, so the objective function doesn't even bother with them.
    """
    agent.assign_parameters(**params)

    # ensure subjective beliefs are used for solution
    if "(Stock)" in agent.name and "Portfolio" in agent.name:
        agent.RiskyAvg = init_subjective_stock["RiskyAvg"]
        agent.RiskyStd = init_subjective_stock["RiskyStd"]
        agent.Rfree = init_subjective_stock["Rfree"]
        agent.update_RiskyDstn()
    if "(Labor)" in agent.name:
        agent.TranShkStd = init_subjective_labor["TranShkStd"]
        agent.PermShkStd = init_subjective_labor["PermShkStd"]
        agent.update_income_process()

    # Solve the model for these parameters, then simulate wealth data
    agent.solve()  # Solve the microeconomic model

    # simulate with true parameters (override subjective beliefs)
    if "(Stock)" in agent.name and "Portfolio" in agent.name:
        agent.RiskyAvg = init_subjective_stock["RiskyAvgTrue"]
        agent.RiskyStd = init_subjective_stock["RiskyStdTrue"]
        agent.Rfree = init_subjective_stock["Rfree"]
        agent.update_RiskyDstn()
    # for labor keep same process as subjective beliefs
    if "(Labor)" in agent.name:
        agent.TranShkStd = init_subjective_labor["TranShkStd"]
        agent.PermShkStd = init_subjective_labor["PermShkStd"]
        agent.update_income_process()

    max_sim_age = agent.T_cycle + 1
    # Initialize the simulation by clearing histories, resetting initial values
    agent.initialize_sim()
    # agent.make_shock_history()
    agent.simulate(max_sim_age)  # Simulate histories of consumption and wealth
    # Take "wealth" to mean bank balances before receiving labor income
    sim_w_history = agent.history["bNrm"]

    # Find the distance between empirical data and simulated medians for each age group

    sim_moments = {
        key: np.median(sim_w_history[cohort_idx])
        for key, cohort_idx in sim_mapping.items()
        if key in emp_moments
    }

    if "Portfolio" in agent.name:
        sim_share_history = agent.history["Share"]
        sim_moments.update(
            {
                key + "_port": np.median(sim_share_history[cohort_idx])
                for key, cohort_idx in sim_mapping.items()
            },
        )

    return sim_moments


def get_moments_cov(agent_name, emp_moments):

    moments_cov = em.get_moments_cov(
        scf_data,
        get_empirical_moments,
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
    params=None,
    subjective_stock=False,
    subjective_labor=False,
):

    agent = make_agent(
        init_agent_name=init_agent_name,
        subjective_stock=subjective_stock,
        subjective_labor=subjective_labor,
    )

    print("Agent created: ", agent.name)

    emp_moments = get_empirical_moments(
        data=scf_data,
        variable="wealth_income_ratio",
        weights="weight",
        groups="age_group",
        mapping=age_mapping,
    )

    if "Portfolio" in agent.name:
        share_moments = get_empirical_moments(
            data=snp_data,
            variable="share",
            groups="age_group",
            mapping=age_mapping,
        )

        suffix = "_port"
        for key, value in share_moments.items():
            emp_moments[key + suffix] = value

    print("Calculated empirical moments.")

    upper_bounds = minimize_options.pop("upper_bounds", None)
    lower_bounds = minimize_options.pop("lower_bounds", None)
    upper_bounds = {
        "CRRA": 20.0,
        "DiscFac": 1.0,
    }
    lower_bounds = {
        "CRRA": 1.1,
        "DiscFac": 0.5,
    }

    moments_cov = get_moments_cov(agent.name, emp_moments)

    print("Calculated moments covariance matrix.")

    initial_guess = get_initial_guess(agent.name, params)

    print("Estimating MSM...")

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

    print("MSM estimation complete.")

    return res


if __name__ == "__main__":
    # Set booleans to determine which tasks should be done
    # Which agent type to estimate ("IndShock" or "Portfolio")
    local_agent_name = "IndShock"

    # Whether to use subjective beliefs
    local_subjective_stock = False
    local_subjective_labor = False

    local_params = {
        "CRRA": 5.0,
        "DiscFac": 0.96,
    }

    estimate_msm(
        init_agent_name=local_agent_name,
        params=local_params,
        subjective_stock=local_subjective_stock,
        subjective_labor=local_subjective_labor,
    )
