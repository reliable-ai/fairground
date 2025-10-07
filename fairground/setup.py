from setuptools import find_packages, setup

setup(
    name="fairground",
    packages=find_packages(),
    version="0.1.0",
    description="A package for managing and processing datasets for fairness research in machine learning",
    author="",
    license="CC BY 4.0",
    install_requires=[
        "pandas",
        "openpyxl",
        "xlrd",
        "air",
        "scikit-learn",
    ],
)
