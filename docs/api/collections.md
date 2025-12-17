# Collections

The `collections` module provides predefined collections of fairness scenarios, developed to maximize the diversity in how existing fair ML methods perform on them.

## Prespecified Collections

::: fairml_datasets.collections.Corpus

::: fairml_datasets.collections.DecorrelatedSmall

::: fairml_datasets.collections.DecorrelatedLarge

::: fairml_datasets.collections.PermissivelyLicensedSmall

::: fairml_datasets.collections.PermissivelyLicensedLarge

::: fairml_datasets.collections.PermissivelyLicensedFull

::: fairml_datasets.collections.GeographicSmall

::: fairml_datasets.collections.GeographicLarge

::: fairml_datasets.collections.GeographicFull

## Usage Examples

### Using the Complete Corpus

```python
from fairml_datasets.collections import Corpus

# Create the corpus (all available datasets and their scenarios)
corpus = Corpus(include_large_datasets=True)

# Iterate through all scenarios in the corpus
for scenario in corpus:
    print(f"Dataset: {scenario.dataset_id}")
    print(f"Sensitive columns: {scenario.sensitive_columns}")
    
    # Load the data
    df = scenario.load(stage="prepared")
```

### Using Predefined Collections

```python
from fairml_datasets.collections import DecorrelatedSmall, PermissivelyLicensedFull, GeographicLarge

# Use a small collection of decorrelated datasets
collection = DecorrelatedSmall()
print(f"Collection contains {len(collection)} scenarios")

# Or use the full collection of permissively licensed datasets
full_collection = PermissivelyLicensedFull()
print(f"Full collection contains {len(full_collection)} scenarios")

# Load and analyze datasets from the geographic collection
geo_collection = GeographicLarge()
for scenario in geo_collection:
    print(f"Dataset: {scenario.dataset_id}")
    print(f"Sensitive columns: {scenario.sensitive_columns}")
    
    # Load the data
    df = scenario.load(stage="prepared")
```
