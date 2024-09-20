import sys

from setuptools import find_packages, setup

sys.path[0:0] = ["estimark"]

setup(
    name="estimark",
    version="0.1.0",
    description="Local package for estimark",
    packages=find_packages(include=["estimark"]),
)
