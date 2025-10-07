# FairML Datasets

A comprehensive Python package for loading, processing, and working with datasets used in fair classification.

## Overview

![The dataset preprocessing pipeline supported by the package.](../assets/pipeline.png)

FairML Datasets provides tools and interfaces to download, load, transform, and analyze the datasets in the *FairGround* corpus. It handles sensitive attributes and facilitates fairness-aware machine learning experiments. The package supports the full data processing pipeline from downloading data all the way to splitting data for ML training.

## Key Features

- 📦 **Loading**: Easily download, load and prepare any of the 44 supported datasets in the corpus.
- 🗂️ **Collections**: Conveneiently use any of our prespecified collections which have been developed to maximize diversity in algorithmic performance.
- 🔄 **Multi-Dataset Support**: Easily evaluate your algorithm on one scenario, five or fourty using a simple loop.
- ⚙️ **Processing**: Automatically apply dataset (pre)processing with configurable choices and defaults available.
- 📊 **Metadata Generation**: Automatically calculate rich metadata features for datasets.
- 💻 **Command-line Interface**: Access common operations without writing code.

## Installation

```bash
pip install fairground
```

Or using uv:

```bash
uv pip install fairground
```

## Quick Start

```python
from fairground import Dataset

# Access a specific dataset by ID directly
dataset = Dataset.from_id("folktables_acsincome_small")

# Load the dataset
df = dataset.load()

# Check sensitive attributes
print(f"Sensitive columns: {dataset.sensitive_columns}")

# Transform the dataset
df_transformed, info = dataset.transform(df)

# Create train/test/validation split
df_train, df_test, df_val = dataset.train_test_val_split(df_transformed)
```

Are you curious which datasets are available? Check out the `Datasets Overview` in the side bar to see the list!

## Command-line Usage

The package provides a command-line interface for common operations:

```bash
# Generate and export metadata
python -m fairground metadata

# Export datasets in various processing stages
python -m fairground export-datasets --stage prepared

# Export dataset citations in BibTeX format
python -m fairground export-citations
```

## Documentation

For detailed API documentation, please see the [API Reference](api/index.md).
