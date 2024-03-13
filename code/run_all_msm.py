from estimark.msm import estimate_msm
from estimark.estimation import get_empirical_moments, get_moments_cov


if __name__ == "__main__":

    emp_moments = get_empirical_moments("")
    port_emp_moments = get_empirical_moments("Porfolio")

    moments_cov = get_moments_cov("", emp_moments)
    port_moments_cov = get_moments_cov("Porfolio", port_emp_moments)

    for i in range(1, 6):
        which_model = str(i)
        for k in range(1, 5):
            subjective_markets = str(k)

            replication_specs = {}

            if which_model == "1" or which_model == "":
                replication_specs["init_agent_name"] = "IndShock"
            elif which_model == "2":
                replication_specs["init_agent_name"] = "Portfolio"
            elif which_model == "3":
                replication_specs["init_agent_name"] = "WarmGlow"
            elif which_model == "4":
                replication_specs["init_agent_name"] = "WarmGlowPortfolio"
            elif which_model == "5":
                replication_specs["init_agent_name"] = "WealthPortfolio"

            print("Model: ", replication_specs["init_agent_name"])

            replication_specs["params_to_estimate"] = ["CRRA", "DiscFac"]

            if subjective_markets == "2" or subjective_markets == "4":
                replication_specs["subjective_stock"] = True
                print("Adding subjective stock market beliefs...")

            if subjective_markets == "3" or subjective_markets == "4":
                replication_specs["subjective_labor"] = True
                print("Adding subjective labor market beliefs...")

            if "Portfolio" in replication_specs["init_agent_name"]:
                replication_specs["emp_moments"] = port_emp_moments
                replication_specs["moments_cov"] = port_moments_cov
            else:
                replication_specs["emp_moments"] = emp_moments
                replication_specs["moments_cov"] = moments_cov

            estimate_msm(**replication_specs)

    print("All replications complete.")
