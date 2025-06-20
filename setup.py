from setuptools import setup, find_packages

setup(
    name="movierec_rl",
    version="0.1.0",
    packages=find_packages(),   # Finds movierec_rl and submodules
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "pandas",
        "scipy",
        "tqdm",
        "pyyaml"
    ],
    python_requires=">=3.8",
)
