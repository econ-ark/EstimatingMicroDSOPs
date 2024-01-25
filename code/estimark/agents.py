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

# Parameters for the consumer type and the estimation

import numpy as np  # Numeric Python
from HARK.ConsumptionSaving.ConsBequestModel import (
    BequestWarmGlowConsumerType,
    BequestWarmGlowPortfolioType,
)

# Import modules from core HARK libraries:
# The consumption-saving micro model
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.ConsPortfolioModel import PortfolioConsumerType
from HARK.ConsumptionSaving.ConsWealthPortfolioModel import WealthPortfolioConsumerType
from HARK.core import AgentType

# Method for sampling from a discrete distribution

# Estimation methods


# Pathnames to the other files:
# Relative directory for primitive parameter files
calibration_dir = "code/calibration/"
# Relative directory for primitive parameter files
tables_dir = "code/tables/"
# Relative directory for primitive parameter files
figures_dir = "content/figures/"
# Relative directory for primitive parameter files
code_dir = "code/"


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


class TempConsumerType(AgentType):
    def __init__(self, cycles=1, **kwds):
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
        super().__init__(cycles=cycles, **kwds)
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
