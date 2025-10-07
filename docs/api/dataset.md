# Dataset

The `Dataset` class is the core component that represents an individual dataset with methods for loading, transforming, and analyzing.

## Class Documentation

::: fairground.dataset.Dataset

## Usage Examples

### Basic Usage

```python
from fairground import Dataset

# Load a dataset directly using Dataset.from_id (recommended)
dataset = Dataset.from_id("folktables_acsincome_small")

# Load the dataset
df = dataset.load()

# Print basic information
print(f"Dataset ID: {dataset.dataset_id}")
print(f"Sensitive columns: {dataset.sensitive_columns}")
print(f"Target column: {dataset.get_target_column()}")
```

### Data Transformation

```python
# Load a dataset
dataset = Dataset.from_id("folktables_acsincome_small")
df = dataset.load()

# Apply standard transformations
df_transformed, info = dataset.transform(df)

# Check transformation info
print(f"Original shape: {df.shape}")
print(f"Transformed shape: {df_transformed.shape}")
print(f"Transformed sensitive columns: {info.sensitive_columns}")
```

### Train/Test Splitting

```python
# Load and transform a dataset
dataset = Dataset.from_id("folktables_acsincome_small")
df = dataset.load()
df_transformed, _ = dataset.transform(df)

# Create train/test/validation split
train, test, val = dataset.train_test_val_split(
    df_transformed, 
    test_size=0.2,
    val_size=0.1,
    random_state=42
)

print(f"Train set size: {len(train)}")
print(f"Test set size: {len(test)}")
print(f"Validation set size: {len(val)}")
```

### Metadata Generation

```python
# Generate metadata for a dataset
dataset = Dataset.from_id("folktables_acsincome_small")
metadata = dataset.generate_metadata()

print(metadata)
```
