"""
Specifies the full set of calibrated values required to estimate the EstimatingMicroDSOPs
model.  The empirical data is stored in a separate csv file and is loaded in setup_scf_data.
"""

import numpy as np
from HARK.Calibration.Income.IncomeTools import CGM_income, parse_income_spec
from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table
from pathlib import Path
from HARK.ConsumptionSaving.ConsIndShockModel import init_lifecycle

# ---------------------------------------------------------------------------------
# Debugging flags
# ---------------------------------------------------------------------------------

show_PermGroFacAgg_error = False
# Error Notes:
# This sets a "quick fix" to the error, AttributeError: 'TempConsumerType' object has no attribute 'PermGroFacAgg'
# If you set this flag to "True" you will see the error. A more thorough fix is to
# fix the place where this error was introduced (Set to "True" and it will appear;
# this was almost certainly introduced when the code was extended to be used in the
# GE setting). An even more thorough solution, which moves beyond the scope of
# fixing this error, is adding unit tests to ID when changes to some code will
# break things elsewhere.
# Note: alternatively, decide that the "init_consumer_objects['PermGroFacAgg'] = 1.0"
# line below fixes it properly ('feature not a bug') and remove all this text.

# ---------------------------------------------------------------------------------
# - Define all of the model parameters for EstimatingMicroDSOPs and ConsumerExamples -
# ---------------------------------------------------------------------------------

exp_nest = 1  # Number of times to "exponentially nest" when constructing a_grid
aXtraMin = 0.001  # Minimum end-of-period "assets above minimum" value
aXtraMax = 100  # Maximum end-of-period "assets above minimum" value
aXtraHuge = None  # A very large value of assets to add to the grid, not used
aXtraExtra = None  # Some other value of assets to add to the grid, not used
aXtraCount = 200  # Number of points in the grid of "assets above minimum"

# Artificial borrowing constraint; imposed minimum level of end-of period assets
BoroCnstArt = 0.0
# Use cubic spline interpolation when True, linear interpolation when False
CubicBool = False
vFuncBool = False  # Whether to calculate the value function during solution

Rfree = 1.03  # Interest factor on assets
# Number of points in discrete approximation to permanent income shocks
PermShkCount = 7
# Number of points in discrete approximation to transitory income shocks
TranShkCount = 7
UnempPrb = 0.05  # Probability of unemployment while working
UnempPrbRet = 0.005  # Probability of "unemployment" while retired
IncUnemp = 0.3  # Unemployment benefits replacement rate
IncUnempRet = 0.0  # "Unemployment" benefits when retired

final_age = 90  # Age at which the problem ends (die with certainty)
retirement_age = 65  # Age at which the consumer retires
initial_age = 25  # Age at which the consumer enters the model
TT = final_age - initial_age  # Total number of periods in the model
retirement_t = retirement_age - initial_age - 1

# Initial guess of the coefficient of relative risk aversion during estimation (rho)
CRRA_start = 5.0
# Initial guess of the adjustment to the discount factor during estimation (beth)
DiscFacAdj_start = 0.99
# Bounds for beth; if violated, objective function returns "penalty value"
DiscFacAdj_bound = [0.0001, 15.0]
# Bounds for rho; if violated, objective function returns "penalty value"
CRRA_bound = [0.0001, 15.0]

# Income
ss_variances = True
income_spec = CGM_income["HS"]
# Replace retirement age
income_spec["age_ret"] = retirement_age
inc_calib = parse_income_spec(
    age_min=initial_age, age_max=final_age, **income_spec, SabelhausSong=ss_variances
)

# Age-varying discount factors over the lifecycle, lifted from Cagetti (2003)


# Get the directory containing the current file and construct the full path to the CSV file
csv_file_path = Path(__file__).resolve().parent / ".." / "data" / "Cagetti2003.csv"
DiscFac_timevary = np.genfromtxt(csv_file_path) * 0.0 + 1.0

# Survival probabilities over the lifecycle
liv_prb = parse_ssa_life_table(
    female=False, min_age=initial_age, max_age=final_age - 1, cohort=1960
)

# Age groups for the estimation: calculate average wealth-to-permanent income ratio
# for consumers within each of these age groups, compare actual to simulated data
empirical_cohort_age_groups = [
    [age for age in range(start, start + 5)] for start in range(26, 61, 5)
]


# Three point discrete distribution of initial w
initial_wealth_income_ratio_vals = np.array([0.17, 0.5, 0.83])
# Equiprobable discrete distribution of initial w
initial_wealth_income_ratio_probs = np.array([0.33333, 0.33333, 0.33334])
num_agents = 10000  # Number of agents to simulate
bootstrap_size = 50  # Number of re-estimations to do during bootstrap
seed = 31382  # Just an integer to seed the estimation

options = {
    "initial_wealth_income_ratio_vals": initial_wealth_income_ratio_vals,
    "initial_wealth_income_ratio_probs": initial_wealth_income_ratio_probs,
    "num_agents": num_agents,
    "bootstrap_size": bootstrap_size,
    "seed": seed,
    "DiscFacAdj_start": DiscFacAdj_start,
    "CRRA_start": CRRA_start,
    "DiscFacAdj_bound": DiscFacAdj_bound,
    "CRRA_bound": CRRA_bound,
    "DiscFac_timevary": DiscFac_timevary,
}

# -----------------------------------------------------------------------------
# -- Set up the dictionary "container" for making a basic lifecycle type ------
# -----------------------------------------------------------------------------

# Dictionary that can be passed to ConsumerType to instantiate
init_consumer_objects = {
    **init_lifecycle,
    **{
        "CRRA": CRRA_start,
        "Rfree": Rfree,
        "PermGroFac": inc_calib["PermGroFac"],
        "BoroCnstArt": BoroCnstArt,
        "PermShkStd": inc_calib["PermShkStd"],
        "PermShkCount": PermShkCount,
        "TranShkStd": inc_calib["TranShkStd"],
        "TranShkCount": TranShkCount,
        "T_cycle": TT,
        "UnempPrb": UnempPrb,
        "UnempPrbRet": UnempPrbRet,
        "T_retire": retirement_t,
        "T_age": TT,
        "IncUnemp": IncUnemp,
        "IncUnempRet": IncUnempRet,
        "aXtraMin": aXtraMin,
        "aXtraMax": aXtraMax,
        "aXtraCount": aXtraCount,
        "aXtraExtra": [aXtraExtra, aXtraHuge],
        "aXtraNestFac": exp_nest,
        "LivPrb": liv_prb,
        "DiscFac": DiscFac_timevary,
        "AgentCount": num_agents,
        "seed": seed,
        "tax_rate": 0.0,
        "vFuncBool": vFuncBool,
        "CubicBool": CubicBool,
    },
}

# from Mateo's JMP for College Educated
ElnR = 0.020
VlnR = 0.424**2

init_subjective_stock_market = {
    "Rfree": 1.019,  # from Mateo's JMP
    "RiskyAvg": np.exp(ElnR + 0.5 * VlnR),
    "RiskyStd": np.sqrt(np.exp(2 * ElnR + VlnR) * (np.exp(VlnR) - 1)),
}

init_subjective_labor_market = {  # from Tao's JMP
    "TranShkStd": [0.03] * len(inc_calib["TranShkStd"]),
    "PermShkStd": [0.03] * len(inc_calib["PermShkStd"]),
}

if show_PermGroFacAgg_error:
    pass  # do nothing
else:
    # print(
    #     "***NOTE: using a 'quick fix' for an attribute error. See 'Error Notes' in EstimationParameter.py for further discussion.***"
    # )
    init_consumer_objects["PermGroFacAgg"] = 1.0

if __name__ == "__main__":
    print("Sorry, estimation_parameters doesn't actually do anything on its own.")
    print("This module is imported by estimation, providing calibrated ")
    print("parameters for the example estimation.  Please see that module if you ")
    print("want more interesting output.")
