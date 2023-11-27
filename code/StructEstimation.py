"""
Demonstrates an example estimation of microeconomic dynamic stochastic optimization
problem, as described in Section 9 of Chris Carroll's SolvingMicroDSOPs.pdf notes.
The estimation attempts to match the age-conditional wealth profile of simulated
consumers to the median wealth holdings of seven age groups in the 2004 SCF by
varying only two parameters: the coefficient of relative risk aversion and a scaling
factor for an age-varying sequence of discount factors.  The estimation uses a
consumption-saving model with idiosyncratic shocks to permanent and transitory
income as defined in ConsIndShockModel.
"""

import csv
import os
import sys
from time import time  # Timing utility

import matplotlib.pyplot as plt
import numpy as np  # Numeric Python
import pylab  # Python reproductions of some Matlab functions
from HARK.ConsumptionSaving.ConsBequestModel import (
    BequestWarmGlowConsumerType,
    BequestWarmGlowPortfolioType,
)

# Import modules from core HARK libraries:
# The consumption-saving micro model
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.ConsPortfolioModel import PortfolioConsumerType

# Method for sampling from a discrete distribution
from HARK.distribution import DiscreteDistribution

# Estimation methods
from HARK.estimation import bootstrap_sample_from_data, minimize_nelder_mead
from scipy.optimize import approx_fprime

# Parameters for the consumer type and the estimation
import Calibration.EstimationParameters as parameters
import Calibration.SetupSCFdata as data  # SCF 2004 data on household wealth
from Code.ConsWealthPortfolioModel import WealthPortfolioConsumerType

# Find pathname to this file:
my_file_path = os.path.dirname(os.path.abspath(__file__))

# Pathnames to the other files:
# Relative directory for primitive parameter files
calibration_dir = os.path.join(my_file_path, "../Calibration/")
# Relative directory for primitive parameter files
tables_dir = os.path.join(my_file_path, "../Tables/")
# Relative directory for primitive parameter files
figures_dir = os.path.join(my_file_path, "../Figures/")
# Relative directory for primitive parameter files
code_dir = os.path.join(my_file_path, "../Code/")

# Add the calibration folder to the path
sys.path.insert(0, os.path.abspath(calibration_dir))

# Need to rely on the manual insertion of pathnames to all files in do_all.py
# NOTE sys.path.insert(0, os.path.abspath(tables_dir)), etc. may need to be
# copied from do_all.py to here


# Set booleans to determine which tasks should be done
# Which agent type to estimate ("IndShock" or "Portfolio")
local_estimation_agent = "IndShock"
local_estimate_model = True  # Whether to estimate the model
# Whether to get standard errors via bootstrap
local_compute_standard_errors = False
# Whether to compute a measure of estimates' sensitivity to moments
local_compute_sensitivity = True
# Whether to make a contour map of the objective function
local_make_contour_plot = True


# =====================================================
# Define objects and functions used for the estimation
# =====================================================


class TempConsumerType:
    def __init__(self, cycles=1, time_flow=True, **kwds):
        """
        Make a new consumer type.

        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.
        time_flow : boolean
            Whether time is currently "flowing" forward for this instance.

        Returns
        -------
        None
        """
        # Initialize a basic AgentType
        super().__init__(cycles=cycles, time_flow=time_flow, **kwds)
        # This estimation uses age-varying discount factors as
        # estimated by Cagetti (2003), so switch from time_inv to time_vary
        self.add_to_time_vary("DiscFac")
        self.del_from_time_inv("DiscFac")

    def check_restrictions(self):
        return None

    def simBirth(self, which_agents):
        """
        Alternate method for simulating initial states for simulated agents, drawing from a finite
        distribution.  Used to overwrite IndShockConsumerType.simBirth, which uses lognormal distributions.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        # Get and store states for newly born agents
        # Take directly from pre-specified distribution
        self.state_now["aNrm"][which_agents] = self.aNrmInit[which_agents]
        # No variation in permanent income needed
        self.state_now["pLvl"][which_agents] = 1.0
        # How many periods since each agent was born
        self.t_age[which_agents] = 0
        # Which period of the cycle each agents is currently in
        self.t_cycle[which_agents] = 0
        return None


class IndShkLifeCycleConsumerType(TempConsumerType, IndShockConsumerType):
    """
    A very lightly edited version of IndShockConsumerType.  Uses an alternate method of making new
    consumers and specifies DiscFac as being age-dependent.  Called "temp" because only used here.
    """


class PortfolioLifeCycleConsumerType(TempConsumerType, PortfolioConsumerType):
    """
    A very lightly edited version of PortfolioConsumerType.  Uses an alternate method of making new
    consumers and specifies DiscFac as being age-dependent.  Called "temp" because only used here.
    """

    def post_solve(self):
        for solution in self.solution:
            solution.cFunc = solution.cFuncAdj
            share = solution.ShareFuncAdj
            solution.ShareFuncAdj = lambda m: np.clip(share(m), 0.0, 1.0)


class BequestWarmGlowLifeCycleConsumerType(
    TempConsumerType, BequestWarmGlowConsumerType
):
    """
    A very lightly edited version of BequestWarmGlowConsumerType.  Uses an alternate method of making new
    consumers and specifies DiscFac as being age-dependent.  Called "temp" because only used here.
    """


class BequestWarmGlowLifeCyclePortfolioType(
    TempConsumerType, BequestWarmGlowPortfolioType
):
    """
    A very lightly edited version of BequestWarmGlowPortfolioType.  Uses an alternate method of making new
    consumers and specifies DiscFac as being age-dependent.  Called "temp" because only used here.
    """

    def post_solve(self):
        for solution in self.solution:
            solution.cFunc = solution.cFuncAdj
            share = solution.ShareFuncAdj
            solution.ShareFuncAdj = lambda m: np.clip(share(m), 0.0, 1.0)


class WealthPortfolioLifeCycleConsumerType(
    TempConsumerType, WealthPortfolioConsumerType
):
    """
    A very lightly edited version of WealthPortfolioConsumerType.  Uses an alternate method of making new
    consumers and specifies DiscFac as being age-dependent.  Called "temp" because only used here.
    """


def weighted_median(values, weights):
    inds = np.argsort(values)
    values = values[inds]
    weights = weights[inds]

    wsum = np.cumsum(inds)
    ind = np.where(wsum > wsum[-1] / 2)[0][0]

    median = values[ind]

    return median


def get_targeted_moments(
    empirical_data=data.w_to_y_data,
    empirical_weights=data.empirical_weights,
    empirical_groups=data.empirical_groups,
    map_simulated_to_empirical_cohorts=data.simulation_map_cohorts_to_age_indices,
):
    # Initialize
    group_count = len(map_simulated_to_empirical_cohorts)
    tgt_moments = np.zeros(group_count)

    for g in range(group_count):
        group_indices = empirical_groups == (g + 1)  # groups are numbered from 1
        tgt_moments[g] = weighted_median(
            empirical_data[group_indices], empirical_weights[group_indices]
        )

    return tgt_moments


# Define the objective function for the simulated method of moments estimation
def simulate_moments(
    DiscFacAdj,
    CRRA,
    agent,
    DiscFacAdj_bound=parameters.DiscFacAdj_bound,
    CRRA_bound=parameters.CRRA_bound,
    map_simulated_to_empirical_cohorts=data.simulation_map_cohorts_to_age_indices,
):
    """
    A quick check to make sure that the parameter values are within bounds.
    Far flung falues of DiscFacAdj or CRRA might cause an error during solution or
    simulation, so the objective function doesn't even bother with them.
    """
    if (
        DiscFacAdj < DiscFacAdj_bound[0]
        or DiscFacAdj > DiscFacAdj_bound[1]
        or CRRA < CRRA_bound[0]
        or CRRA > CRRA_bound[1]
    ):
        return 1e30 * np.ones(len(map_simulated_to_empirical_cohorts))

    # Update the agent with a new path of DiscFac based on this DiscFacAdj (and a new CRRA)
    agent.DiscFac = [b * DiscFacAdj for b in parameters.DiscFac_timevary]
    agent.CRRA = CRRA
    # Solve the model for these parameters, then simulate wealth data
    agent.solve()  # Solve the microeconomic model
    # "Unpack" the consumption function for convenient access
    # agent.unpack("cFunc")
    max_sim_age = max([max(ages) for ages in map_simulated_to_empirical_cohorts]) + 1
    # Initialize the simulation by clearing histories, resetting initial values
    agent.initialize_sim()
    agent.simulate(max_sim_age)  # Simulate histories of consumption and wealth
    # Take "wealth" to mean bank balances before receiving labor income
    sim_w_history = agent.history["bNrm"]

    # Find the distance between empirical data and simulated medians for each age group
    group_count = len(map_simulated_to_empirical_cohorts)
    sim_moments = []
    for g in range(group_count):
        # The simulated time indices corresponding to this age group
        cohort_indices = map_simulated_to_empirical_cohorts[g]
        # The median of simulated wealth-to-income for this age group
        sim_moments += [np.median(sim_w_history[cohort_indices,])]

    sim_moments = np.array(sim_moments)

    return sim_moments


def smmObjectiveFxn(
    DiscFacAdj,
    CRRA,
    agent,
    tgt_moments,
    DiscFacAdj_bound=parameters.DiscFacAdj_bound,
    CRRA_bound=parameters.CRRA_bound,
    map_simulated_to_empirical_cohorts=data.simulation_map_cohorts_to_age_indices,
):
    """
    The objective function for the SMM estimation.  Given values of discount factor
    adjuster DiscFacAdj, coeffecient of relative risk aversion CRRA, a base consumer
    agent type, empirical data, and calibrated parameters, this function calculates
    the weighted distance between data and the simulated wealth-to-permanent
    income ratio.

    Steps:
        a) solve for consumption functions for (DiscFacAdj, CRRA)
        b) simulate wealth holdings for many consumers over time
        c) sum distances between empirical data and simulated medians within
            seven age groupings

    Parameters
    ----------
    DiscFacAdj : float
        An adjustment factor to a given age-varying sequence of discount factors.
        I.e. DiscFac[t] = DiscFacAdj*DiscFac_timevary[t].
    CRRA : float
        Coefficient of relative risk aversion.
    agent : ConsumerType
        The consumer type to be used in the estimation, with all necessary para-
        meters defined except the discount factor and CRRA.
    DiscFacAdj_bound : (float,float)
        Lower and upper bounds on DiscFacAdj; if outside these bounds, the function
        simply returns a "penalty value".
    DiscFacAdj_bound : (float,float)
        Lower and upper bounds on CRRA; if outside these bounds, the function
        simply returns a "penalty value".
    empirical_data : np.array
        Array of wealth-to-permanent-income ratios in the data.
    empirical_weights : np.array
        Weights for each observation in empirical_data.
    empirical_groups : np.array
        Array of integers listing the age group for each observation in empirical_data.
    map_simulated_to_empirical_cohorts : [np.array]
        List of arrays of "simulation ages" for each age grouping.  E.g. if the
        0th element is [1,2,3,4,5], then these time indices from the simulation
        correspond to the 0th empirical age group.

    Returns
    -------
    distance_sum : float
        Sum of distances between empirical data observations and the corresponding
        median wealth-to-permanent-income ratio in the simulation.
    """

    sim_moments = simulate_moments(
        DiscFacAdj,
        CRRA,
        agent,
        DiscFacAdj_bound,
        CRRA_bound,
        map_simulated_to_empirical_cohorts,
    )
    errors = tgt_moments - sim_moments
    loss = np.dot(errors, errors)

    return loss


# Define the bootstrap procedure
def calculateStandardErrorsByBootstrap(
    initial_estimate, N, agent, seed=0, verbose=False
):
    """
    Calculates standard errors by repeatedly re-estimating the model with datasets
    resampled from the actual data.

    Parameters
    ----------
    initial_estimate : [float,float]
        The estimated [DiscFacAdj,CRRA], for use as an initial guess for each
        re-estimation in the bootstrap procedure.
    N : int
        Number of times to resample data and re-estimate the model.
    seed : int
        Seed for the random number generator.
    verbose : boolean
        Indicator for whether extra output should be printed for the user.

    Returns
    -------
    standard_errors : [float,float]
        Standard errors calculated by bootstrap: [DiscFacAdj_std_error, CRRA_std_error].
    """
    t_0 = time()

    # Generate a list of seeds for generating bootstrap samples
    RNG = np.random.default_rng(seed)
    seed_list = RNG.integers(2**31 - 1, size=N)

    # Estimate the model N times, recording each set of estimated parameters
    estimate_list = []
    for n in range(N):
        t_start = time()

        # Bootstrap a new dataset by resampling from the original data
        bootstrap_data = (
            bootstrap_sample_from_data(data.scf_data_array, seed=seed_list[n])
        ).T
        w_to_y_data_bootstrap = bootstrap_data[0,]
        empirical_groups_bootstrap = bootstrap_data[1,]
        empirical_weights_bootstrap = bootstrap_data[2,]

        # Find moments with bootstrapped sample
        bstrap_tgt_moments = get_targeted_moments(
            empirical_data=w_to_y_data_bootstrap,
            empirical_weights=empirical_weights_bootstrap,
            empirical_groups=empirical_groups_bootstrap,
            map_simulated_to_empirical_cohorts=data.simulation_map_cohorts_to_age_indices,
        )

        # Make a temporary function for use in this estimation run
        def smmObjectiveFxnBootstrap(parameters_to_estimate):
            return smmObjectiveFxn(
                DiscFacAdj=parameters_to_estimate[0],
                CRRA=parameters_to_estimate[1],
                agent=agent,
                tgt_moments=bstrap_tgt_moments,
                map_simulated_to_empirical_cohorts=data.simulation_map_cohorts_to_age_indices,
            )

        # Estimate the model with the bootstrap data and add to list of estimates
        this_estimate = minimize_nelder_mead(smmObjectiveFxnBootstrap, initial_estimate)
        estimate_list.append(this_estimate)
        t_now = time()

        # Report progress of the bootstrap
    if verbose:
        print(
            f"Finished bootstrap estimation #{n + 1} of {N} in {t_now - t_start} seconds ({t_now - t_0} cumulative)"
        )

    # Calculate the standard errors for each parameter
    estimate_array = (np.array(estimate_list)).T
    DiscFacAdj_std_error = np.std(estimate_array[0])
    CRRA_std_error = np.std(estimate_array[1])

    return [DiscFacAdj_std_error, CRRA_std_error]


# =================================================================
# Done defining objects and functions.  Now run them (if desired).
# =================================================================


def estimate(
    estimation_agent=local_estimation_agent,
    estimate_model=local_estimate_model,
    compute_standard_errors=local_compute_standard_errors,
    compute_sensitivity=local_compute_sensitivity,
    make_contour_plot=local_make_contour_plot,
):
    """
    Run the main estimation procedure for SolvingMicroDSOP.

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

    if estimation_agent == "IndShock":
        agent_type = IndShkLifeCycleConsumerType
    elif estimation_agent == "Portfolio":
        agent_type = PortfolioLifeCycleConsumerType
    elif estimation_agent == "WarmGlow":
        agent_type = BequestWarmGlowLifeCycleConsumerType
    elif estimation_agent == "WarmGlowPortfolio":
        agent_type = BequestWarmGlowLifeCyclePortfolioType
    elif estimation_agent == "WealthPortfolio":
        agent_type = WealthPortfolioLifeCycleConsumerType

    # Make a lifecycle consumer to be used for estimation, including simulated
    # shocks (plus an initial distribution of wealth)
    # Make a TempConsumerType for estimation
    EstimationAgent = agent_type(**parameters.init_consumer_objects)
    # Set the number of periods to simulate
    EstimationAgent.T_sim = EstimationAgent.T_cycle + 1
    # Choose to track bank balances as wealth
    EstimationAgent.track_vars = ["bNrm"]
    # Draw initial assets for each consumer
    EstimationAgent.aNrmInit = DiscreteDistribution(
        parameters.initial_wealth_income_ratio_probs,
        parameters.initial_wealth_income_ratio_vals,
        seed=parameters.seed,
    ).draw(N=parameters.num_agents)
    EstimationAgent.make_shock_history()

    targeted_moments = get_targeted_moments()

    # Estimate the model using Nelder-Mead
    if estimate_model:
        initial_guess = [parameters.DiscFacAdj_start, parameters.CRRA_start]
        print("----------------------------------------------------------------------")
        print(
            f"Now estimating the model using Nelder-Mead from an initial guess of {initial_guess}..."
        )
        print("----------------------------------------------------------------------")

        # Make a single-input lambda function for use in the optimizer
        def smmObjectiveFxnReduced(parameters_to_estimate):
            """
            A "reduced form" of the SMM objective function, compatible with the optimizer.
            Identical to smmObjectiveFunction, but takes only a single input as a length-2
            list representing [DiscFacAdj,CRRA].
            """
            return smmObjectiveFxn(
                DiscFacAdj=parameters_to_estimate[0],
                CRRA=parameters_to_estimate[1],
                agent=EstimationAgent,
                tgt_moments=targeted_moments,
            )

        t_start_estimate = time()
        model_estimate = minimize_nelder_mead(
            smmObjectiveFxnReduced, initial_guess, verbose=True
        )
        t_end_estimate = time()

        time_to_estimate = t_end_estimate - t_start_estimate
        print(
            f"Time to execute all: {round(time_to_estimate / 60.0, 2)} min, {time_to_estimate} sec"
        )
        print(
            f"Estimated values: DiscFacAdj={model_estimate[0]}, CRRA={model_estimate[1]}"
        )

        # Create the simple estimate table
        estimate_results_file = os.path.join(
            tables_dir, estimation_agent + "_estimate_results.csv"
        )
        with open(estimate_results_file, "wt") as f:
            writer = csv.writer(f)
            writer.writerow(["DiscFacAdj", "CRRA"])
            writer.writerow([model_estimate[0], model_estimate[1]])

    if compute_standard_errors and not estimate_model:
        print(
            "To run the bootstrap you must first estimate the model by setting estimate_model = True."
        )

    # Compute standard errors by bootstrap
    if compute_standard_errors and estimate_model:
        # Estimate the model:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            f"Computing standard errors using {parameters.bootstrap_size} bootstrap replications."
        )
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        try:
            t_bootstrap_guess = time_to_estimate * parameters.bootstrap_size
            minutes, seconds = divmod(t_bootstrap_guess, 60)
            print(f"This will take approximately {minutes:.2f} min, {seconds:.2f} sec")

        except:
            pass
        t_start_bootstrap = time()
        std_errors = calculateStandardErrorsByBootstrap(
            model_estimate,
            N=parameters.bootstrap_size,
            agent=EstimationAgent,
            seed=parameters.seed,
            verbose=True,
        )
        t_end_bootstrap = time()
        time_to_bootstrap = t_end_bootstrap - t_start_bootstrap
        print(
            f"Time to execute all: {time_to_bootstrap / 60:.2f} min, {time_to_bootstrap:.2f} sec"
        )
        print(
            f"Standard errors: DiscFacAdj--> {std_errors[0]}, CRRA--> {std_errors[1]}"
        )

        # Create the simple bootstrap table
        bootstrap_results_file = os.path.join(
            tables_dir, estimation_agent + "_bootstrap_results.csv"
        )
        with open(bootstrap_results_file, "wt") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "DiscFacAdj",
                    "DiscFacAdj_standard_error",
                    "CRRA",
                    "CRRA_standard_error",
                ]
            )
            writer.writerow(
                [model_estimate[0], std_errors[0], model_estimate[1], std_errors[1]]
            )

    # Compute sensitivity measure
    if compute_sensitivity and estimate_model:
        print("``````````````````````````````````````````````````````````````````````")
        print("Computing sensitivity measure.")
        print("``````````````````````````````````````````````````````````````````````")

        # Find the Jacobian of the function that simulates moments
        def simulate_moments_reduced(x):
            moments = simulate_moments(
                x[0],
                x[1],
                agent=EstimationAgent,
                DiscFacAdj_bound=parameters.DiscFacAdj_bound,
                CRRA_bound=parameters.CRRA_bound,
                map_simulated_to_empirical_cohorts=data.simulation_map_cohorts_to_age_indices,
            )

            return moments

        n_moments = len(data.simulation_map_cohorts_to_age_indices)
        jac = np.array(
            [
                approx_fprime(
                    model_estimate,
                    lambda x: simulate_moments_reduced(x)[j],
                    epsilon=0.01,
                )
                for j in range(n_moments)
            ]
        )

        # Compute sensitivity measure. (all moments weighted equally)
        sensitivity = np.dot(np.linalg.inv(np.dot(jac.T, jac)), jac.T)

        # Create lables for moments in the plots
        moment_labels = [
            "[" + str(min(x)) + "," + str(max(x)) + "]"
            for x in data.empirical_cohort_age_groups
        ]

        # Plot
        fig, axs = plt.subplots(len(initial_guess))
        fig.set_tight_layout(True)

        axs[0].bar(range(n_moments), sensitivity[0, :], tick_label=moment_labels)
        axs[0].set_title("DiscFacAdj")
        axs[0].set_ylabel("Sensitivity")
        axs[0].set_xlabel("Median W/Y Ratio")

        axs[1].bar(range(n_moments), sensitivity[1, :], tick_label=moment_labels)
        axs[1].set_title("CRRA")
        axs[1].set_ylabel("Sensitivity")
        axs[1].set_xlabel("Median W/Y Ratio")

        plt.savefig(os.path.join(figures_dir, estimation_agent + "Sensitivity.pdf"))
        plt.savefig(os.path.join(figures_dir, estimation_agent + "Sensitivity.png"))
        plt.savefig(os.path.join(figures_dir, estimation_agent + "Sensitivity.svg"))

        plt.show()

    # Make a contour plot of the objective function
    if make_contour_plot:
        print("``````````````````````````````````````````````````````````````````````")
        print("Creating the contour plot.")
        print("``````````````````````````````````````````````````````````````````````")
        t_start_contour = time()
        DiscFac_star, CRRA_star = model_estimate
        grid_density = 20  # Number of parameter values in each dimension
        level_count = 100  # Number of contour levels to plot
        DiscFacAdj_list = np.linspace(
            max(DiscFac_star - 0.25, 0.5), min(DiscFac_star + 0.25, 1.05), grid_density
        )
        CRRA_list = np.linspace(
            max(CRRA_star - 5, 2), min(CRRA_star + 5, 8), grid_density
        )
        CRRA_mesh, DiscFacAdj_mesh = pylab.meshgrid(CRRA_list, DiscFacAdj_list)
        smm_obj_levels = np.empty([grid_density, grid_density])
        for j in range(grid_density):
            DiscFacAdj = DiscFacAdj_list[j]
            for k in range(grid_density):
                CRRA = CRRA_list[k]
                smm_obj_levels[j, k] = smmObjectiveFxn(
                    DiscFacAdj,
                    CRRA,
                    agent=EstimationAgent,
                    tgt_moments=targeted_moments,
                )
        smm_contour = pylab.contourf(
            CRRA_mesh, DiscFacAdj_mesh, smm_obj_levels, level_count
        )
        t_end_contour = time()
        time_to_contour = t_end_contour - t_start_contour
        print(
            f"Time to execute all: {time_to_contour / 60:.2f} min, {time_to_contour:.2f} sec"
        )
        pylab.colorbar(smm_contour)
        pylab.plot(model_estimate[1], model_estimate[0], "*r", ms=15)
        pylab.xlabel(r"coefficient of relative risk aversion $\rho$", fontsize=14)
        pylab.ylabel(r"discount factor adjustment $\beth$", fontsize=14)
        pylab.savefig(os.path.join(figures_dir, estimation_agent + "SMMcontour.pdf"))
        pylab.savefig(os.path.join(figures_dir, estimation_agent + "SMMcontour.png"))
        pylab.savefig(os.path.join(figures_dir, estimation_agent + "SMMcontour.svg"))
        pylab.show()


def estimate_all():
    agent_types = [
        PortfolioLifeCycleConsumerType,
        BequestWarmGlowLifeCycleConsumerType,
        BequestWarmGlowLifeCyclePortfolioType,
        WealthPortfolioLifeCycleConsumerType,
    ]

    agent_names = [
        "(a) Portfolio Choice",
        "(b) Separable Wealth in Utility",
        "(c) Portfolio Separable Wealth in Utility",
        "(d) Non-Separable Portfolio Wealth in Utility",
    ]

    fig_sensitivity = plt.figure(layout="constrained", figsize=(14, 12))
    fig_sensitivity.suptitle("Sensitivity of Moments to Parameters")

    subfigs_sensitivity = fig_sensitivity.subfigures(2, 2)

    fig_contour = plt.figure(layout="constrained", figsize=(14, 12))
    fig_contour.suptitle("Contour Plots")

    subfigs_contour = fig_contour.subplots(2, 2)

    for count, agent_type in enumerate(agent_types):
        # Make a lifecycle consumer to be used for estimation, including simulated
        # shocks (plus an initial distribution of wealth)
        # Make a TempConsumerType for estimation
        EstimationAgent = agent_type(**parameters.init_consumer_objects)
        # Set the number of periods to simulate
        EstimationAgent.T_sim = EstimationAgent.T_cycle + 1
        # Choose to track bank balances as wealth
        EstimationAgent.track_vars = ["bNrm"]
        # Draw initial assets for each consumer
        EstimationAgent.aNrmInit = DiscreteDistribution(
            parameters.initial_wealth_income_ratio_probs,
            parameters.initial_wealth_income_ratio_vals,
            seed=parameters.seed,
        ).draw(N=parameters.num_agents)
        EstimationAgent.make_shock_history()

        targeted_moments = get_targeted_moments()

        idx = np.unravel_index(count, (2, 2))

        initial_guess = [parameters.DiscFacAdj_start, parameters.CRRA_start]
        print("----------------------------------------------------------------------")
        print(
            f"Now estimating the model using Nelder-Mead from an initial guess of {initial_guess}..."
        )
        print("----------------------------------------------------------------------")

        # Make a single-input lambda function for use in the optimizer
        def smmObjectiveFxnReduced(parameters_to_estimate):
            """
            A "reduced form" of the SMM objective function, compatible with the optimizer.
            Identical to smmObjectiveFunction, but takes only a single input as a length-2
            list representing [DiscFacAdj,CRRA].
            """
            return smmObjectiveFxn(
                DiscFacAdj=parameters_to_estimate[0],
                CRRA=parameters_to_estimate[1],
                agent=EstimationAgent,
                tgt_moments=targeted_moments,
            )

        t_start_estimate = time()
        model_estimate = minimize_nelder_mead(
            smmObjectiveFxnReduced, initial_guess, verbose=True
        )
        t_end_estimate = time()

        time_to_estimate = t_end_estimate - t_start_estimate
        print(
            f"Time to execute all: {round(time_to_estimate / 60.0, 2)} min, {time_to_estimate} sec"
        )
        print(
            f"Estimated values: DiscFacAdj={model_estimate[0]}, CRRA={model_estimate[1]}"
        )

        print("``````````````````````````````````````````````````````````````````````")
        print("Computing sensitivity measure.")
        print("``````````````````````````````````````````````````````````````````````")

        # Find the Jacobian of the function that simulates moments
        def simulate_moments_reduced(x):
            moments = simulate_moments(
                x[0],
                x[1],
                agent=EstimationAgent,
                DiscFacAdj_bound=parameters.DiscFacAdj_bound,
                CRRA_bound=parameters.CRRA_bound,
                map_simulated_to_empirical_cohorts=data.simulation_map_cohorts_to_age_indices,
            )

            return moments

        n_moments = len(data.simulation_map_cohorts_to_age_indices)
        jac = np.array(
            [
                approx_fprime(
                    model_estimate,
                    lambda x: simulate_moments_reduced(x)[j],
                    epsilon=0.01,
                )
                for j in range(n_moments)
            ]
        )

        # Compute sensitivity measure. (all moments weighted equally)
        sensitivity = np.dot(np.linalg.inv(np.dot(jac.T, jac)), jac.T)

        # Create lables for moments in the plots
        moment_labels = [
            "[" + str(min(x)) + "," + str(max(x)) + "]"
            for x in data.empirical_cohort_age_groups
        ]

        # Plot
        subfigs_sensitivity[idx].suptitle(agent_names[count])
        subfigsnest_sensitivity = subfigs_sensitivity[idx].subplots(2, 1)

        subfigsnest_sensitivity[0].bar(
            range(n_moments), sensitivity[0, :], tick_label=moment_labels
        )
        subfigsnest_sensitivity[0].set_title("DiscFacAdj")
        subfigsnest_sensitivity[0].set_ylabel("Sensitivity")
        subfigsnest_sensitivity[0].set_xlabel("Median W/Y Ratio")

        subfigsnest_sensitivity[1].bar(
            range(n_moments), sensitivity[1, :], tick_label=moment_labels
        )
        subfigsnest_sensitivity[1].set_title("CRRA")
        subfigsnest_sensitivity[1].set_ylabel("Sensitivity")
        subfigsnest_sensitivity[1].set_xlabel("Median W/Y Ratio")

        print("``````````````````````````````````````````````````````````````````````")
        print("Creating the contour plot.")
        print("``````````````````````````````````````````````````````````````````````")
        t_start_contour = time()
        DiscFac_star, CRRA_star = model_estimate
        grid_density = 20  # Number of parameter values in each dimension
        level_count = 100  # Number of contour levels to plot
        DiscFacAdj_list = np.linspace(
            max(DiscFac_star - 0.25, 0.5), min(DiscFac_star + 0.25, 1.05), grid_density
        )
        CRRA_list = np.linspace(
            max(CRRA_star - 5, 2), min(CRRA_star + 5, 8), grid_density
        )
        CRRA_mesh, DiscFacAdj_mesh = pylab.meshgrid(CRRA_list, DiscFacAdj_list)
        smm_obj_levels = np.empty([grid_density, grid_density])
        for j in range(grid_density):
            DiscFacAdj = DiscFacAdj_list[j]
            for k in range(grid_density):
                CRRA = CRRA_list[k]
                smm_obj_levels[j, k] = smmObjectiveFxn(
                    DiscFacAdj,
                    CRRA,
                    agent=EstimationAgent,
                    tgt_moments=targeted_moments,
                )

        # Create figure and axes objects

        # Plot the contour
        subfigs_contour[idx].set_title(agent_names[count])
        contour = subfigs_contour[idx].contourf(
            CRRA_mesh, DiscFacAdj_mesh, smm_obj_levels, level_count
        )
        cbar = fig_sensitivity.colorbar(contour)

        # Plot the model estimate
        subfigs_contour[idx].plot(model_estimate[1], model_estimate[0], "*r", ms=15)

        # Set axis labels and title
        subfigs_contour[idx].set_xlabel(
            r"coefficient of relative risk aversion $\rho$", fontsize=14
        )
        subfigs_contour[idx].set_ylabel(
            r"discount factor adjustment $\beth$", fontsize=14
        )

        # Print time to execute
        t_end_contour = time()
        time_to_contour = t_end_contour - t_start_contour
        print(
            f"Time to execute all: {time_to_contour / 60:.2f} min, {time_to_contour:.2f} sec"
        )

    fig_sensitivity.savefig(os.path.join(figures_dir, "AllSensitivity.pdf"))
    fig_sensitivity.savefig(os.path.join(figures_dir, "AllSensitivity.png"))
    fig_sensitivity.savefig(os.path.join(figures_dir, "AllSensitivity.svg"))

    # Save and show the plot
    fig_contour.savefig(os.path.join(figures_dir, "AllSMMcontour.pdf"))
    fig_contour.savefig(os.path.join(figures_dir, "AllSMMcontour.png"))
    fig_contour.savefig(os.path.join(figures_dir, "AllSMMcontour.svg"))


if __name__ == "__main__":
    estimate()