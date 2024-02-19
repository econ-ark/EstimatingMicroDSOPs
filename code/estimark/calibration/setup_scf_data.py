"""
Sets up the SCF data for use in the EstimatingMicroDSOPs estimation.
"""

from estimark.calibration.estimation_parameters import (
    empirical_cohort_age_groups,
    initial_age,
)

import numpy as np  # Numerical Python
import pandas as pd


from pathlib import Path

# Get the directory containing the current file and construct the full path to the CSV file
csv_file_path = Path(__file__).resolve().parent / ".." / "data" / "SCFdata.csv"

# Read the CSV file
scf_data = pd.read_csv(csv_file_path)


# Keep only observations with normal incomes > 0,
# otherwise wealth/income is not well defined
scf_data = scf_data[scf_data.norminc > 0.0]

# Initialize empty lists for the data
w_to_y_data = []  # Ratio of wealth to permanent income
empirical_weights = []  # Weighting for this observation
empirical_groups = []  # Which age group this observation belongs to (1-7)

# Only extract the data required from the SCF dataset
search_ages = [ages[-1] for ages in empirical_cohort_age_groups]
for idx, age in enumerate(search_ages):
    age_data = scf_data[scf_data.age_group.str.contains(f"{age}]")]
    w_to_y_data.append(age_data.wealth_income_ratio.values)
    empirical_weights.append(age_data.weight.values)
    # create a group id 1-7 for every observation
    empirical_groups.append([idx + 1] * len(age_data))

# Convert SCF data to numpy's array format for easier math
w_to_y_data = np.concatenate(w_to_y_data)
empirical_weights = np.concatenate(empirical_weights)
empirical_groups = np.concatenate(empirical_groups)

# Generate a single array of SCF data, useful for resampling for bootstrap
scf_data_array = np.array([w_to_y_data, empirical_groups, empirical_weights]).T

# Generate a mapping between the real ages in the groups and the indices of simulated data
simulation_map_cohorts_to_age_indices = []
for ages in empirical_cohort_age_groups:
    simulation_map_cohorts_to_age_indices.append(np.array(ages) - initial_age)

if __name__ == "__main__":
    print("Sorry, setup_scf_data doesn't actually do anything on its own.")
    print("This module is imported by estimation, providing data for")
    print("the example estimation.  Please see that module if you want more")
    print("interesting output.")
