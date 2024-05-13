from estimark.estimation import estimate
from estimark.options import low_resource

agent_names = [
    "Portfolio",
    "WarmGlowPortfolio",
    "WealthPortfolio",
]


# Ask the user which replication to run, and run it:
def run_replication():
    for agent_name in agent_names:

        replication_specs = low_resource.copy()
        replication_specs["agent_name"] = agent_name
        replication_specs["save_dir"] = "content/tables/TRP"

        print("Model: ", replication_specs["agent_name"])

        estimate(**replication_specs)

    print("All replications complete.")


if __name__ == "__main__":
    run_replication()
