"""
Specifies the full set of calibrated values required to estimate the EstimatingMicroDSOPs
model.  The empirical data is stored in a separate csv file and is loaded in setup_scf_data.
"""

from pathlib import Path

import numpy as np
from HARK.Calibration.Income.IncomeTools import CGM_income, parse_income_spec
from HARK.ConsumptionSaving.ConsIndShockModel import init_lifecycle
from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table

# ---------------------------------------------------------------------------------
# - Define all of the model parameters for EstimatingMicroDSOPs and ConsumerExamples -
# ---------------------------------------------------------------------------------

exp_nest = 1  # Number of times to "exponentially nest" when constructing a_grid
aXtraMin = 0.001  # Minimum end-of-period "assets above minimum" value
aXtraMax = 100  # Maximum end-of-period "assets above minimum" value
aXtraCount = 100  # Number of points in the grid of "assets above minimum"

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
terminal_t = final_age - initial_age  # Total number of periods in the model
retirement_t = retirement_age - initial_age - 1

# Initial guess of the coefficient of relative risk aversion during estimation (rho)
init_CRRA = 5.0
# Initial guess of the adjustment to the discount factor during estimation (beth)
init_DiscFacAdj = 0.99
# Bounds for beth; if violated, objective function returns "penalty value"
bounds_DiscFacAdj = [0.0001, 15.0]
# Bounds for rho; if violated, objective function returns "penalty value"
bounds_CRRA = [0.0001, 15.0]

# Income
ss_variances = True
income_spec = CGM_income["HS"]
# Replace retirement age
income_spec["age_ret"] = retirement_age
inc_calib = parse_income_spec(
    age_min=initial_age,
    age_max=final_age,
    **income_spec,
    SabelhausSong=ss_variances,
)

# Age-varying discount factors over the lifecycle, lifted from Cagetti (2003)
# Get the directory containing the current file and construct the full path to the CSV file
csv_file_path = Path(__file__).resolve().parent / ".." / "data" / "Cagetti2003.csv"
timevary_DiscFac = np.genfromtxt(csv_file_path) * 0.0 + 1.0  # todo
constant_DiscFac = np.ones_like(timevary_DiscFac)

# Survival probabilities over the lifecycle
liv_prb = parse_ssa_life_table(
    female=False,
    min_age=initial_age,
    max_age=final_age - 1,
    cohort=1960,
)


# Three point discrete distribution of initial w
init_w_to_y = np.array([0.17, 0.5, 0.83])
# Equiprobable discrete distribution of initial w
prob_w_to_y = np.array([0.33333, 0.33333, 0.33334])
num_agents = 10000  # Number of agents to simulate
bootstrap_size = 50  # Number of re-estimations to do during bootstrap
seed = 31382  # Just an integer to seed the estimation

options = {
    "init_w_to_y": init_w_to_y,
    "prob_w_to_y": prob_w_to_y,
    "num_agents": num_agents,
    "bootstrap_size": bootstrap_size,
    "seed": seed,
    "init_DiscFacAdj": init_DiscFacAdj,
    "init_CRRA": init_CRRA,
    "bounds_DiscFacAdj": bounds_DiscFacAdj,
    "bounds_CRRA": bounds_CRRA,
    "timevary_DiscFac": timevary_DiscFac,
}

# -----------------------------------------------------------------------------
# -- Set up the dictionary "container" for making a basic lifecycle type ------
# -----------------------------------------------------------------------------

# Dictionary that can be passed to ConsumerType to instantiate
init_consumer_objects = {
    **init_lifecycle,
    **{
        "CRRA": init_CRRA,
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
        "DiscFac": timevary_DiscFac,
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

TrueElnR = 0.085
TrueVlnR = 0.170**2

init_subjective_stock = {
    "Rfree": 1.019,  # from Mateo's JMP
    "RiskyAvg": np.exp(ElnR + 0.5 * VlnR),
    "RiskyStd": np.sqrt(np.exp(2 * ElnR + VlnR) * (np.exp(VlnR) - 1)),
    "RiskyAvgTrue": np.exp(TrueElnR + 0.5 * TrueVlnR),
    "RiskyStdTrue": np.sqrt(np.exp(2 * TrueElnR + TrueVlnR) * (np.exp(TrueVlnR) - 1)),
}

init_subjective_labor = {  # from Tao's JMP
    "TranShkStd": [0.03] * len(inc_calib["TranShkStd"]),
    "PermShkStd": [0.03] * len(inc_calib["PermShkStd"]),
}


if __name__ == "__main__":
    print("Sorry, estimation_parameters doesn't actually do anything on its own.")
    print("This module is imported by estimation, providing calibrated ")
    print("parameters for the example estimation.  Please see that module if you ")
    print("want more interesting output.")
