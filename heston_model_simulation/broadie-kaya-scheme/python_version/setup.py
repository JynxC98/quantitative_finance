from setuptools import setup, find_packages

setup(
    name="bk_helpers",
    version="0.1.0",
    description="Broadie-Kaya exact simulation helpers for the Heston model",
    author="Harsh Parikh",
    packages=find_packages(include=["bk_helpers", "bk_helpers.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "numba",
    ],
)
