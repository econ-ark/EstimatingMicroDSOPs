"""Sets up the SCF data for use in the EstimatingMicroDSOPs estimation.
"""

from pathlib import Path

import numpy as np  # Numerical Python
import pandas as pd

from estimark.parameters import final_age_data, initial_age

# Get the directory containing the current file and construct the full path to the CSV file
csv_file_path = Path(__file__).resolve().parent / ".." / "data" / "SCFdata.csv"

# Read the CSV file
scf_full_data = pd.read_csv(csv_file_path)


# Keep only observations with normal incomes > 0,
# otherwise wealth/income is not well defined
scf_full_data = scf_full_data[scf_full_data.norminc > 0.0]

# Age groups for the estimation: calculate average wealth-to-permanent income ratio
# for consumers within each of these age groups, compare actual to simulated data
age_groups = [
    list(range(start, start + 5))
    for start in range(initial_age + 1, final_age_data + 1, 5)
]

# Initialize empty lists for the data
scf_data = []  # Ratio of wealth to permanent income
scf_weights = []  # Weighting for this observation
scf_groups = []  # Which age group this observation belongs to (1-7)

# Only extract the data required from the SCF dataset
search_ages = [ages[-1] for ages in age_groups]
for idx, age in enumerate(search_ages):
    age_data = scf_full_data[scf_full_data.age_group.str.contains(f"{age}]")]
    scf_data.append(age_data.wealth_income_ratio.values)
    scf_weights.append(age_data.weight.values)
    # create a group id 1-7 for every observation
    scf_groups.append([idx + 1] * len(age_data))

# Convert SCF data to numpy's array format for easier math
scf_data = np.concatenate(scf_data)
scf_weights = np.concatenate(scf_weights)
scf_groups = np.concatenate(scf_groups)

# Generate a single array of SCF data, useful for resampling for bootstrap
scf_array = np.array([scf_data, scf_groups, scf_weights]).T

# Generate a mapping between the real ages in the groups and the indices of simulated data
scf_mapping = []
for ages in age_groups:
    scf_mapping.append(np.array(ages) - initial_age)

if __name__ == "__main__":
    print("Sorry, setup_scf_data doesn't actually do anything on its own.")
    print("This module is imported by estimation, providing data for")
    print("the example estimation.  Please see that module if you want more")
    print("interesting output.")
