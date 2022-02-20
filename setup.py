from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="t4clab",
    version="0.1.0",
    author="",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # PyTorch etc.
        "torch",
        "pytorch-lightning",
        "torchmetrics",
        # General science & ml
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "tables",
        # Plotting & visualization
        "jupyterlab",
        "ipympl",
        "matplotlib",
        "seaborn",
        # hydra & logging
        "hydra-core",
        "hydra-experiment-sweeper",
        "hydra-submitit-launcher",
        "omegaconf",
        "wandb",
        # Utilities
        "tqdm",
        "rich",
        "ipython",
        "ipdb",
    ],
    tests_require=["pytest"],
    classifiers=[],
)
