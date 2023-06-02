"""
Run all of the plots and tables in SolvingMicroDSOPs.

To execute, do the following on the Python command line:

    from HARK.[YOUR-MODULE-NAME-HERE].do_all import run_replication
    run_replication()

You will be presented with an interactive prompt that asks what level of
replication you would like to have.

More Details
------------

This example script allows the user to create all of the Figures and Tables
modules for SolvingMicroDSOPs.StructuralEstimation.

This is example is kept as simple and minimal as possible to illustrate the
format of a "replication archive."

The file structure is as follows:

./SolvingMicroDSOPs/
    Calibration/        # Directory that contain the necessary code and data to parameterize the model
    Code/               # The main estimation code, in this case StructuralEstimation.py
    Figures/            # Any Figures created by the main code
    Tables/             # Any tables created by the main code

Because computational modeling can be very memory- and time-intensive, this file
also allows the user to choose whether to run files based on there resouce
requirements. Files are categorized as one of the following three:

- low_resource:     low RAM needed and runs quickly, say less than 1-5 minutes
- medium_resource:  moderate RAM needed and runs moderately quickly, say 5-10+ mintues
- high_resource:    high RAM needed (and potentially parallel computing required), and high time to run, perhaps even hours, days, or longer.

The designation is purposefully vague and left up the to researcher to specify
more clearly below. Using time taken on an example machine is entirely reasonable
here.

Finally, this code may serve as example code for efforts that fall outside
the HARK package structure for one reason or another. Therefore this script will
attempt to import the necessary MicroDSOP sub-modules as though they are part of
the HARK package; if that fails, this script reverts to manaully updating the
Python PATH with the locations of the MicroDSOP directory structure so it can
still run.
"""


import os
import sys

from Code.StructEstimation import estimate
from Calibration.Options import (
    low_resource,
    medium_resource,
    high_resource,
    all_replications,
)

# Find pathname to this file:
my_file_path = os.path.dirname(os.path.abspath(__file__))

# Pathnames to the other files:
# Relative directory for primitive parameter files
calibration_dir = os.path.join(my_file_path, "Calibration")
# Relative directory for primitive parameter files
tables_dir = os.path.join(my_file_path, "Tables")
# Relative directory for primitive parameter files
figures_dir = os.path.join(my_file_path, "Figures")
# Relative directory for primitive parameter files
code_dir = os.path.join(my_file_path, "Code")


# manually add the pathnames to the various files directly to the beginning
# of the Python path. This will be needed for all files that will run in
# lower directories.
sys.path.insert(0, calibration_dir)
sys.path.insert(0, tables_dir)
sys.path.insert(0, figures_dir)
sys.path.insert(0, code_dir)
sys.path.insert(0, my_file_path)

# Manual import needed, should draw from first instance at start of Python
# PATH added above:


# Ask the user which replication to run, and run it:
def run_replication():
    which_model = input(
        """Which model would you like to run?
        
        [1] IndShockConsumerType
        
        2   PortfolioConsumerType \n\n"""
    )

    which_replication = input(
        """Which replication would you like to run? (See documentation in do_all.py for details.) Please enter the option number to run that option; default is in brackets:

        [1] low-resource:    ~90 sec; output ./Tables/estimate_results.csv

         2  medium-resource: ~7 min;  output ./Figures/SMMcontour.pdf
                                             ./Figures/SMMcontour.png
         3  high-resource:   ~30 min; output ./Tables/bootstrap_results.csv

         4  all:             ~40 min; output: all above.

         q  quit: exit without executing.\n\n"""
    )

    replication_specs = {}

    if which_model == "1" or which_model == "":
        replication_specs["estimation_agent"] = "IndShock"
    elif which_model == "2":
        replication_specs["estimation_agent"] = "Portfolio"

    if which_replication == "q":
        return

    elif which_replication == "1" or which_replication == "":
        print("Running low-resource replication...")
        replication_specs.update(**low_resource)

    elif which_replication == "2":
        print("Running medium-resource replication...")
        replication_specs.update(**medium_resource)

    elif which_replication == "3":
        print("Running high-resource replication...")
        replication_specs.update(**high_resource)

    elif which_replication == "4":
        print("Running all replications...")
        replication_specs.update(**all_replications)

    else:
        return

    print(replication_specs)
    estimate(**replication_specs)


if __name__ == "__main__":
    run_replication()
