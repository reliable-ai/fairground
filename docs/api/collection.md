# Collection

The `Collection` classes provide a way to organize and iterate through groups of related fairness scenarios, making it easier to run batch analyses across multiple datasets with different sensitive attribute configurations.

## Class Documentation

::: fairground.collection.Collection

::: fairground.collection.PrespecifiedCollection

## Usage Examples

### Basic Usage

```python
from fairground.scenario import Scenario
from fairground.collection import Collection

# Create scenarios
scenario1 = Scenario("adult", ["sex"])
scenario2 = Scenario("compas", ["race"])

# Create a collection
collection = Collection([scenario1, scenario2])

# Iterate through scenarios
for scenario in collection:
    print(f"Dataset: {scenario.dataset_id}, Sensitive columns: {scenario.sensitive_columns}")

    df = scenario.load(stage="prepared")
```
