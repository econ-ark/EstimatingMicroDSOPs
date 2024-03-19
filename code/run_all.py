import itertools

import dask
from dask.distributed import Client
from estimark.estimation import estimate
from estimark.options import low_resource

agent_names = [
    "IndShock",
    "Portfolio",
    "WarmGlow",
    "WarmGlowPortfolio",
    "WealthPortfolio",
]


# Ask the user which replication to run, and run it:
def run_replication():
    client = Client(threads_per_worker=10, n_workers=20)

    lazy_results = []

    for agent_name in agent_names:
        for sub_stock, sub_labor in itertools.product(range(2), repeat=2):
            temp_agent_name = agent_name
            if sub_stock or sub_labor:
                temp_agent_name += "Sub"
                if sub_stock:
                    temp_agent_name += "(Stock)"
                if sub_labor:
                    temp_agent_name += "(Labor)"
                temp_agent_name += "Market"

            replication_specs = low_resource.copy()
            replication_specs["agent_name"] = temp_agent_name
            replication_specs["save_dir"] = "content/tables/min"

            print("Model: ", replication_specs["agent_name"])

            lazy_result = dask.delayed(estimate)(**replication_specs)
            lazy_results.append(lazy_result)

    dask.compute(*lazy_results)

    client.close()

    print("All replications complete.")


if __name__ == "__main__":
    run_replication()
