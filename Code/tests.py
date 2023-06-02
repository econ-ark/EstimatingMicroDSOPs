from do_all import low_resource, medium_resource, struct


def test_low_resource():
    print("Running low-resource replication...")
    struct.main(**low_resource)


def test_medium_resource():
    print("Running medium-resource replication...")
    struct.main(**medium_resource)
