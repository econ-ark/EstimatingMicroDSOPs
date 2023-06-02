from Code.StructEstimation import estimate
from Calibration.Options import low_resource, medium_resource


def test_low_resource():
    print("Running low-resource replication...")
    estimate(**low_resource)


def test_medium_resource():
    print("Running medium-resource replication...")
    estimate(**medium_resource)

def test_portfolio_medium_resource():
    print("Running medium-resource replication...")
    estimate(**medium_resource, estimation_agent="Portfolio")
