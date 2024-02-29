from pathlib import Path

import numpy as np  # Numerical Python
import pandas as pd

from estimark.parameters import final_age_data, initial_age

file_path = (
    Path(__file__).resolve().parent / ".." / "data" / "S&P Target Date glidepath.xlsx"
)

# Load data
snp = pd.read_excel(file_path)

# Filter data using loc
snp = snp.loc[
    (snp["Current Age"] >= initial_age) & (snp["Current Age"] <= final_age_data)
]

# Create age groups and code NaNs as 0
bins = range(initial_age, final_age_data + 1, 5)
labels = np.arange(1, len(bins))
snp["age_groups"] = pd.cut(snp["Current Age"], bins=bins, labels=labels)

# Get targeted moments
