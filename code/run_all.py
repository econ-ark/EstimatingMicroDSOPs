import dask
from dask.distributed import Client
from estimark.estimation import estimate
from estimark.options import (
    all_replications,
    high_resource,
    low_resource,
    medium_resource,
)


# Ask the user which replication to run, and run it:
def run_replication():
    client = Client(threads_per_worker=10, n_workers=20)

    lazy_results = []

    for i in range(1, 6):
        which_model = str(i)
        which_replication = "1"
        for k in range(1, 5):
            subjective_markets = str(k)

            replication_specs = {}

            if which_model == "1" or which_model == "":
                replication_specs["agent_name"] = "IndShock"
            elif which_model == "2":
                replication_specs["agent_name"] = "Portfolio"
            elif which_model == "3":
                replication_specs["agent_name"] = "WarmGlow"
            elif which_model == "4":
                replication_specs["agent_name"] = "WarmGlowPortfolio"
            elif which_model == "5":
                replication_specs["agent_name"] = "WealthPortfolio"
            else:
                print("Invalid model choice.")
                return

            print("Model: ", replication_specs["agent_name"])

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
                print("Invalid replication choice.")
                return

            if subjective_markets == "2" or subjective_markets == "4":
                replication_specs["subjective_stock"] = True
                print("Adding subjective stock market beliefs...")

            if subjective_markets == "3" or subjective_markets == "4":
                replication_specs["subjective_labor"] = True
                print("Adding subjective labor market beliefs...")

            lazy_result = dask.delayed(estimate)(**replication_specs)
            lazy_results.append(lazy_result)

    dask.compute(*lazy_results)

    client.close()

    print("All replications complete.")


if __name__ == "__main__":
    run_replication()
