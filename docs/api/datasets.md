# Datasets

The `Datasets` class provides a collection of datasets with methods for filtering and batch operations. It serves as an alternative entry point for accessing individual datasets in the package.

## Class Documentation

::: fairground.datasets.Datasets

## Usage Examples

### Loading All Datasets

```python
from fairground import Datasets

# Load all available datasets
datasets = Datasets()

# Print information about available datasets
print(f"Number of datasets: {len(datasets)}")
print(f"Available dataset IDs: {[dataset.dataset_id for dataset in datasets]}")
```
