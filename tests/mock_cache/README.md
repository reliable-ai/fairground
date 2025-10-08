# Mock Cache

The mock cache contains smaller versions of existing datasets for testing purposes.

`folktables_acsincome_small.parquet` was created with the following snippet to get 5 rows for each stratum.

```python
import pandas as pd
df = pd.read_parquet("cache/datasets/folktables_acsincome_small.parquet")
df.groupby(['PINCP', 'RAC1P']).sample(5, random_state=80539).to_parquet("folktables_acsincome_small.parquet")
```
