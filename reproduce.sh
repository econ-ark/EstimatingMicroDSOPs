# Check if the environment exists before creating it
if ! conda env list | grep -q 'solvingmicrodsops'; then
    mamba env create -qq -f environment.yml
fi

# Activate the environment
conda activate solvingmicrodsops

# Execute script to reproduce figures
python code/do_all.py
