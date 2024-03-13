"""Specifies the full set of calibrated values required to estimate the EstimatingMicroDSOPs
model.  The empirical data is stored in a separate csv file and is loaded in setup_scf_data.
"""

import numpy as np
from HARK.Calibration.Income.IncomeTools import CGM_income, parse_income_spec
from HARK.ConsumptionSaving.ConsIndShockModel import init_lifecycle
from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table
from HARK.distribution import DiscreteDistribution

# ---------------------------------------------------------------------------------
# - Define all of the model parameters for EstimatingMicroDSOPs and ConsumerExamples -
# ---------------------------------------------------------------------------------

# Assets grid
exp_nest = 1  # Number of times to "exponentially nest" when constructing a_grid
aXtraMin = 0.001  # Minimum end-of-period "assets above minimum" value
aXtraMax = 100  # Maximum end-of-period "assets above minimum" value
aXtraCount = 20  # Number of points in the grid of "assets above minimum"

# Artificial borrowing constraint
BoroCnstArt = 0.0  # imposed minimum level of end-of period assets
Rfree = 1.03  # Interest factor on assets

# Use cubic spline interpolation when True, linear interpolation when False
CubicBool = False
vFuncBool = False  # Whether to calculate the value function during solution

# Income process parameters
# Number of points in discrete approximation to permanent income shocks
PermShkCount = 7
# Number of points in discrete approximation to transitory income shocks
TranShkCount = 7
UnempPrb = 0.05  # Probability of unemployment while working
UnempPrbRet = 0.005  # Probability of "unemployment" while retired # maybe one more zero
IncUnemp = 0.3  # Unemployment benefits replacement rate
IncUnempRet = 0.0  # "Unemployment" benefits when retired
ss_variances = True  # Use the Sabelhaus-Song variance profiles
education = "College"  # Education level for income process

# Population age parameters
final_age = 120  # Age at which the problem ends (die with certainty)
retirement_age = 65  # Age at which the consumer retires
initial_age = 25  # Age at which the consumer enters the model
final_age_data = 95  # Age at which the data ends
age_interval = 5  # Interval between age groups

# Three point discrete distribution of initial w
init_w_to_y = np.array([0.17, 0.5, 0.83])
# Equiprobable discrete distribution of initial w
prob_w_to_y = np.array([0.33333, 0.33333, 0.33334])
num_agents = 10000  # Number of agents to simulate

# Bootstrap options
bootstrap_size = 50  # Number of re-estimations to do during bootstrap
seed = 1132023  # Just an integer to seed the estimation


params_to_estimate = ["CRRA", "DiscFac"]
# Initial guess of the coefficient of relative risk aversion during estimation (rho)
init_CRRA = 5.0
# Initial guess of the adjustment to the discount factor during estimation (beth)
init_DiscFac = 0.95
# Bounds for beth; if violated, objective function returns "penalty value"
bounds_DiscFac = [0.5, 1.0]
# Bounds for rho; if violated, objective function returns "penalty value"
bounds_CRRA = [1.1, 20.0]


######################################################################
# Constructed parameters
######################################################################

# Total number of periods in the model
terminal_t = final_age - initial_age
retirement_t = retirement_age - initial_age - 1

# Income
income_spec = CGM_income[education]
# Replace retirement age
income_spec["age_ret"] = retirement_age
inc_calib = parse_income_spec(
    age_min=initial_age,
    age_max=final_age,
    **income_spec,
    SabelhausSong=ss_variances,
)

# Age groups for the estimation: calculate average wealth-to-permanent income ratio
# for consumers within each of these age groups, compare actual to simulated data

age_groups = [
    list(range(start, start + age_interval))
    for start in range(initial_age + 1, final_age_data + 1, age_interval)
]

# generate labels as (25,30], (30,35], ...
age_labels = [f"({group[0]-1},{group[-1]}]" for group in age_groups]

# Generate mappings between the real ages in the groups and the indices of simulated data
age_mapping = dict(zip(age_labels, map(np.array, age_groups)))
sim_mapping = {
    label: np.array(group) - initial_age for label, group in zip(age_labels, age_groups)
}

remove_ages_from_scf = np.arange(
    retirement_age - age_interval + 1,
    retirement_age + age_interval + 1,
)  # remove retirement ages 61-70
remove_ages_from_snp = np.arange(
    retirement_age + age_interval + 1,
)  # only match ages 71 and older


init_params_options = {
    "init_guess": {"CRRA": init_CRRA, "DiscFac": init_DiscFac},
    "upper_bounds": {"CRRA": bounds_CRRA[1], "DiscFac": bounds_DiscFac[1]},
    "lower_bounds": {"CRRA": bounds_CRRA[0], "DiscFac": bounds_DiscFac[0]},
}


# Survival probabilities over the lifecycle
liv_prb = parse_ssa_life_table(
    female=False,
    min_age=initial_age,
    max_age=final_age - 1,
    cohort=1960,
)

aNrmInit = DiscreteDistribution(
    prob_w_to_y,
    init_w_to_y,
    seed=seed,
).draw(N=num_agents)

bootstrap_options = {
    "bootstrap_size": bootstrap_size,
    "seed": seed,
}

minimize_options = {
    "algorithm": "scipy_neldermead",
    "multistart": True,
    "error_handling": "continue",
    "algo_options": {
        "convergence.absolute_params_tolerance": 1e-3,
        "convergence.absolute_criterion_tolerance": 1e-3,
        "stopping.max_iterations": 50,
        "stopping.max_criterion_evaluations": 100,
        # "n_cores": 12,
    },
    # "numdiff_options": {"n_cores": 12},
}

# -----------------------------------------------------------------------------
# -- Set up the dictionary "container" for making a basic lifecycle type ------
# -----------------------------------------------------------------------------

# Dictionary that can be passed to ConsumerType to instantiate
init_calibration = {
    **init_lifecycle,
    "CRRA": init_CRRA,
    "DiscFac": init_DiscFac,
    "Rfree": Rfree,
    "PermGroFac": inc_calib["PermGroFac"],
    "PermGroFacAgg": 1.0,
    "BoroCnstArt": BoroCnstArt,
    "PermShkStd": inc_calib["PermShkStd"],
    "PermShkCount": PermShkCount,
    "TranShkStd": inc_calib["TranShkStd"],
    "TranShkCount": TranShkCount,
    "T_cycle": terminal_t,
    "UnempPrb": UnempPrb,
    "UnempPrbRet": UnempPrbRet,
    "T_retire": retirement_t,
    "T_age": terminal_t,
    "IncUnemp": IncUnemp,
    "IncUnempRet": IncUnempRet,
    "aXtraMin": aXtraMin,
    "aXtraMax": aXtraMax,
    "aXtraCount": aXtraCount,
    "aXtraNestFac": exp_nest,
    "LivPrb": liv_prb,
    "AgentCount": num_agents,
    "seed": seed,
    "tax_rate": 0.0,
    "vFuncBool": vFuncBool,
    "CubicBool": CubicBool,
    "aNrmInit": aNrmInit,
}

# from Mateo's JMP for College Educated
ElnR = 0.020
VlnR = 0.424**2

TrueElnR = 0.085
TrueVlnR = 0.170**2

init_subjective_stock = {
    "Rfree": 1.019,  # from Mateo's JMP
    "RiskyAvg": np.exp(ElnR + 0.5 * VlnR),
    "RiskyStd": np.sqrt(np.exp(2 * ElnR + VlnR) * (np.exp(VlnR) - 1)),
    "RiskyAvgTrue": np.exp(TrueElnR + 0.5 * TrueVlnR),
    "RiskyStdTrue": np.sqrt(np.exp(2 * TrueElnR + TrueVlnR) * (np.exp(TrueVlnR) - 1)),
}

# from Tao's JMP
init_subjective_labor = {
    "TranShkStd": [0.03] * len(inc_calib["TranShkStd"]),
    "PermShkStd": [0.03] * len(inc_calib["PermShkStd"]),
}
