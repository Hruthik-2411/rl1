from setuptools import find_packages, setup

setup(
    name="faultdiag",
    version="0.1.0",
    description="Cost-aware active fault diagnosis and recovery with RL under partial observability",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.10",
)
