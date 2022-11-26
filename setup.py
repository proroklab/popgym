from setuptools import find_packages, setup

setup(
    name="popgym",
    version="0.0.1",
    description="A collection of partially-observable procedural gym environments",
    python_requires=">=3.6",
    install_requires=["gym==0.24.0", "numpy", "mazelib"],
    # Add "ray[rllib]" when they fix their gym dep
    extras_require={"baselines": ["torch", "opt_einsum", "wandb", "dnc", "einops"]},
)
