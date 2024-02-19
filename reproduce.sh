# Check if the environment exists before creating it
if ! conda env list | grep -q 'estimatingmicrodsops'; then
    mamba env create -qq -f environment.yml
fi

# Activate the environment
conda activate estimatingmicrodsops

# Execute script to reproduce figures
python code/do_all.py
