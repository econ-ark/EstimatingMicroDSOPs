# Define settings for "main()" function in StructuralEstiamtion.py based on
# resource requirements:

params_to_estimate = ["CRRA"]

low_resource = {
    "estimate_model": True,
    "params_to_estimate": params_to_estimate,
    "make_contour_plot": False,
    "compute_se_bootstrap": False,
    "compute_sensitivity": False,
    "subjective_stock": False,
    "subjective_labor": False,
}
# Author note:
# This takes approximately 90 seconds on a laptop with the following specs:
# Linux, Ubuntu 14.04.1 LTS, 8G of RAM, Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz

medium_resource = {
    "estimate_model": True,
    "params_to_estimate": params_to_estimate,
    "make_contour_plot": True,
    "compute_se_bootstrap": False,
    "compute_sensitivity": True,
    "subjective_stock": False,
    "subjective_labor": False,
}
# Author note:
# This takes approximately 7 minutes on a laptop with the following specs:
# Linux, Ubuntu 14.04.1 LTS, 8G of RAM, Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz

high_resource = {
    "estimate_model": True,
    "params_to_estimate": params_to_estimate,
    "make_contour_plot": False,
    "compute_se_bootstrap": True,
    "compute_sensitivity": True,
    "subjective_stock": False,
    "subjective_labor": False,
}
# Author note:
# This takes approximately 30 minutes on a laptop with the following specs:
# Linux, Ubuntu 14.04.1 LTS, 8G of RAM, Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz

all_replications = {
    "estimate_model": True,
    "params_to_estimate": params_to_estimate,
    "make_contour_plot": True,
    "compute_se_bootstrap": True,
    "compute_sensitivity": True,
    "subjective_stock": False,
    "subjective_labor": False,
}
# Author note:
# This takes approximately 40 minutes on a laptop with the following specs:
# Linux, Ubuntu 14.04.1 LTS, 8G of RAM, Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz
