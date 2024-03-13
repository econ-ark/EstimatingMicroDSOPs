from estimark.estimation import (
    do_compute_se_boostrap,
    do_compute_sensitivity,
    do_estimate_model,
    do_make_contour_plot,
    get_empirical_moments,
    get_initial_guess,
    make_agent,
)
from estimark.parameters import bootstrap_options, minimize_options
from pathlib import Path


# Pathnames to the other files:
# Relative directory for primitive parameter files
tables_dir = "content/tables/min/"
Path(tables_dir).mkdir(parents=True, exist_ok=True)
# Relative directory for primitive parameter files
figures_dir = "content/figures/min/"
Path(figures_dir).mkdir(parents=True, exist_ok=True)


def estimate_min(
    init_agent_name,
    params_to_estimate,
    estimate_model=True,
    compute_se_bootstrap=False,
    compute_sensitivity=False,
    make_contour_plot=False,
    subjective_stock=False,
    subjective_labor=False,
):
    """Run the main estimation procedure for SolvingMicroDSOP.

    Parameters
    ----------
    estimate_model : bool
        Whether to estimate the model using Nelder-Mead. When True, this is a low-time, low-memory operation.

    compute_standard_errors : bool
        Whether to compute standard errors on the estiamtion of the model.

    make_contour_plot : bool
        Whether to make the contour plot associate with the estiamte.

    Returns
    -------
    None

    """
    ############################################################
    # Make agent
    ############################################################

    agent = make_agent(
        init_agent_name=init_agent_name,
        subjective_stock=subjective_stock,
        subjective_labor=subjective_labor,
    )

    ############################################################
    # Get empirical moments
    ############################################################

    emp_moments = get_empirical_moments(agent.name)

    ############################################################
    # Get initial guess
    ############################################################

    initial_guess = get_initial_guess(
        agent.name,
        params_to_estimate,
        tables_dir,
    )

    ############################################################
    # Estimate model
    ############################################################

    if estimate_model:
        model_estimate, time_to_estimate = do_estimate_model(
            agent,
            emp_moments,
            initial_guess,
            minimize_options=minimize_options,
            tables_dir=tables_dir,
        )

        # Compute standard errors by bootstrap
        if compute_se_bootstrap:
            do_compute_se_boostrap(
                agent,
                model_estimate,
                time_to_estimate,
                tables_dir=tables_dir,
                **bootstrap_options,
            )

        # Compute sensitivity measure
        if compute_sensitivity:
            do_compute_sensitivity(
                agent,
                model_estimate,
                initial_guess,
                figures_dir=figures_dir,
            )

        # Make a contour plot of the objective function
        if make_contour_plot:
            do_make_contour_plot(
                agent,
                model_estimate,
                emp_moments,
                figures_dir=figures_dir,
            )


if __name__ == "__main__":
    # Set booleans to determine which tasks should be done
    # Which agent type to estimate ("IndShock" or "Portfolio")
    local_agent_name = "Portfolio"
    local_params_to_estimate = ["CRRA", "DiscFac"]
    local_estimate_model = True  # Whether to estimate the model
    # Whether to get standard errors via bootstrap
    local_compute_se_bootstrap = False
    # Whether to compute a measure of estimates' sensitivity to moments
    local_compute_sensitivity = False
    # Whether to make a contour map of the objective function
    local_make_contour_plot = False
    # Whether to use subjective beliefs
    local_subjective_stock = False
    local_subjective_labor = False

    estimate_min(
        init_agent_name=local_agent_name,
        params_to_estimate=local_params_to_estimate,
        estimate_model=local_estimate_model,
        compute_se_bootstrap=local_compute_se_bootstrap,
        compute_sensitivity=local_compute_sensitivity,
        make_contour_plot=local_make_contour_plot,
        subjective_stock=local_subjective_stock,
        subjective_labor=local_subjective_labor,
    )
