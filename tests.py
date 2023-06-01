from do_all import low_resource, struct


def test_low_resource():
    print("Running low-resource replication...")
    struct.main(**low_resource)
