<script setup>
import { computed } from "vue";
import { data as collectionsData } from "../data/collections.data.js";
import { data as metadata } from "../data/metadata.data.js";
import HighlightedCode from "./HighlightedCode.vue";

const props = defineProps({
  name: {
    type: String,
    required: true,
  },
  filename: {
    type: String,
    required: true,
  },
  badgeField: {
    type: String,
    default: "ann_domain_class",
  },
});

const collection = computed(() => {
  return collectionsData[props.filename] || [];
});

const datasets = computed(() => {
  if (!collection.value || collection.value.length === 0) return [];

  // Parse scenario IDs and extract dataset IDs and sensitive attributes
  return collection.value
    .map((item) => {
      if (!item.scenario_id) return null;

      // Parse scenario_id format: dataset_id:sensitive_attributes
      const parts = item.scenario_id.split(":");
      const datasetId = parts[0];

      // Get sensitive attributes (could be multiple, separated by semicolons)
      const sensitiveAttrs = parts.length > 1 ? parts[1] : "";

      // Find dataset in metadata
      const datasetInfo = metadata.find(
        (d) => d.ann_new_dataset_id === datasetId,
      );

      // Create a base object with essential fields
      const result = {
        ...datasetInfo,

        id: datasetId,
        sensitiveAttributes: sensitiveAttrs
          .split(";")
          .map((attr) => attr.trim())
          .filter((attr) => attr),
        name: datasetInfo?.ann_dataset_name || datasetId,
      };
      return result;
    })
    .filter(Boolean);
});

const exampleCode = computed(() => {
  const collectionName = props.name.replace(/[^a-zA-Z0-9]/g, "");
  return `from fairml_datasets.collections import ${collectionName}

collection = ${collectionName}() # [!code highlight]

# The collection consists of scenarios
for scenario in collection:
    # Each scenario behaves just like a dataset

    # Load as pandas DataFrame
    df = scenario.load()  # or df = scenario.to_pandas()
    print(f"Dataset shape: {df.shape}")

    # Get the target column
    target_column = scenario.get_target_column()
    print(f"Target column: {target_column}")

    # Get sensitive attributes (before transformation)
    sensitive_columns_org = scenario.sensitive_columns

    # Transform to  e.g. impute missing data
    df_transformed, transformation_info = scenario.transform(df)
    # Sensitive columns may change due to transformation
    sensitive_columns = transformation_info['sensitive_columns']

    # Split into train and test sets
    train_df, test_df = scenario.train_test_split(df, test_size=0.3);

    # Run analyses on the data`;
});
</script>

<template>
  <div class="collection-details">
    <div v-if="datasets.length" class="collection-intro">
      <p>This collection contains <b>{{ datasets.length }}</b> scenarios (each scenario is a combination of a dataset with a particular selection of sensitive attributes).</p>
    </div>

    <div v-if="datasets.length === 0" class="collection-empty">
      <p>No datasets found in this collection.</p>
    </div>

    <div class="datasets-list">
      <a v-for="(dataset, index) in datasets"
         :key="dataset.id"
         class="dataset-item"
         :href="`/datasets/${dataset.id}`">
        <div class="dataset-main-info">
          <h3>
            <span class="dataset-number">{{ index + 1 }}.</span>
            <span class="dataset-name">{{ dataset.name }}</span>
          </h3>
          <span v-if="dataset[props.badgeField]" class="field-badge">{{ dataset[props.badgeField] }}</span>
        </div>

        <div class="dataset-meta">
          <span class="meta-item sensitive-attrs">
            <span v-for="(attr, index) in dataset.sensitiveAttributes" :key="index" class="sensitive-attr">
              {{ attr }}
            </span>
            <span v-if="dataset.sensitiveAttributes.length === 0" class="no-attrs">No sensitive attributes</span>
          </span>
        </div>
      </a>
    </div>

    <div class="example-code">
      <h2>Example Code</h2>
      Example code showing how to use this collection in Python:
      <HighlightedCode :code="exampleCode" language="python" />
    </div>
  </div>
</template>

<style scoped>
.collection-details {
  margin: 2rem 0;
}

.collection-intro {
  margin-bottom: 1.5rem;
}

.collection-empty {
  padding: 2rem;
  background-color: var(--vp-c-bg-soft);
  border-radius: 8px;
  text-align: center;
  color: var(--vp-c-text-2);
}

.datasets-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.dataset-item {
  background-color: var(--vp-c-bg-soft);
  border-radius: 8px;
  padding: 1rem 1.5rem;
  display: flex;
  flex-direction: column;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  color: var(--vp-c-text-1);
  text-decoration: none;
}

.dataset-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
}

.dataset-main-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.dataset-main-info h3 {
  margin: 0;
  font-size: 1.1rem;
}

.dataset-number {
  font-weight: 500;
  color: var(--vp-c-text-2);
  margin-right: 0.3rem;
}

.dataset-name {
  color: var(--vp-c-brand);
}

.field-badge {
  background-color: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-dark);
  font-size: 0.8rem;
  padding: 0.2rem 0.5rem;
  border-radius: 16px;
  white-space: nowrap;
}

.dataset-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  font-size: 0.9rem;
}

.meta-item {
  display: flex;
  align-items: center;
}

.meta-label {
  font-weight: 500;
  color: var(--vp-c-text-2);
  margin-right: 0.25rem;
}

.meta-value {
  color: var(--vp-c-text-1);
}

.sensitive-attrs {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
}

.sensitive-attr {
  background-color: var(--vp-c-gray-soft);
  color: var(--vp-c-gray-dark);
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
  font-size: 0.8rem;
}

.no-attrs {
  color: var(--vp-c-text-2);
  font-style: italic;
  font-size: 0.8rem;
}

@media (max-width: 640px) {
  .dataset-main-info {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }

  .field-badge {
    align-self: flex-start;
  }

  .dataset-meta {
    flex-direction: column;
    gap: 0.5rem;
  }
}
</style>