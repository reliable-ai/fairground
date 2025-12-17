<script setup>
import { computed, ref } from "vue";
import { data as metadata } from "../data/metadata.data.js";
import HighlightedCode from "./HighlightedCode.vue";

const props = defineProps({
  id: {
    type: String,
    required: true,
  },
});

const activeTab = ref("overview");
const codeCopied = ref(false);

const dataset = computed(() => {
  return metadata.find((d) => d.ann_new_dataset_id === props.id) || {};
});

const relatedDatasets = computed(() => {
  if (!dataset.value.ann_dataset_base_id) return [];

  return metadata.filter(
    (d) =>
      d.ann_base_dataset_name === dataset.value.ann_base_dataset_name &&
      d.ann_new_dataset_id !== dataset.value.ann_new_dataset_id,
  );
});

const pythonExample = computed(() => {
  return `from fairml_datasets import Dataset

# Get the dataset
dataset = Dataset.from_id("${props.id}") # [!code highlight]

# Load as pandas DataFrame
df = dataset.load()  # or df = dataset.to_pandas()
print(f"Dataset shape: {df.shape}")

# Get the target column
target_column = dataset.get_target_column()
print(f"Target column: {target_column}")

# Get sensitive attributes (before transformation)
sensitive_columns_org = dataset.sensitive_columns

# Transform to e.g. impute missing data
df_transformed, transformation_info = dataset.transform(df)
# Sensitive columns may change due to transformation
sensitive_columns = transformation_info.sensitive_columns

# Split into train and test sets
train_df, test_df = dataset.train_test_split(df, test_size=0.3)

# Do amazing analyses ✨️
`;
});

const cliExample = computed(() => {
  return `python fairml_datasets export-datasets --id=${props.id}

# Or via uv
uv run fairml_datasets export-datasets --id=${props.id}`;
});

const handleCodeCopy = () => {
  codeCopied.value = true;

  setTimeout(() => {
    codeCopied.value = false;
  }, 2000);
};
</script>

<template>
  <div v-if="dataset.ann_new_dataset_id" class="dataset-details">
    <div class="dataset-header">
      <div v-if="dataset.ann_dataset_aliases" class="dataset-aliases">
        Also known as: {{ dataset.ann_dataset_aliases }}
      </div>
    </div>

    <!-- Warning for dataset annotation -->
    <div v-if="dataset.ann_warning" class="dataset-warning warning custom-block mb-4">
      <strong>⚠️ Warning:</strong> {{ dataset.ann_warning }}
    </div>
    <!-- Warning for missing license -->
    <div v-if="!dataset.ann_license || dataset.ann_license === 'not found'" class="dataset-warning warning custom-block mb-4">
      <strong>⚠️ Warning:</strong> This dataset is missing license information.
    </div>

    <div class="dataset-tabs">
      <button
        :class="{ active: activeTab === 'overview' }"
        @click="activeTab = 'overview'">Overview</button>
      <button
        :class="{ active: activeTab === 'details' }"
        @click="activeTab = 'details'">Details</button>
      <button
        :class="{ active: activeTab === 'classification' }"
        @click="activeTab = 'classification'">Classification Task</button>
      <button
        :class="{ active: activeTab === 'fairness' }"
        @click="activeTab = 'fairness'">Fairness Info</button>
      <button
        :class="{ active: activeTab === 'sources' }"
        @click="activeTab = 'sources'">Sources</button>
    </div>

    <!-- Overview Tab -->
    <div v-show="activeTab === 'overview'" class="tab-content">
      <div class="dataset-section">
        <div class="dataset-badges">
          <span class="badge domain">{{ dataset.ann_domain_class || 'Unknown Domain' }}</span>
          <span v-if="dataset.ann_data_type" class="badge data-type">{{ dataset.ann_data_type }}</span>
          <span v-if="dataset.ann_high_priority" class="badge priority">High Priority</span>
          <span v-if="dataset.ann_is_large" class="badge large">Large Dataset</span>
        </div>

        <div class="info-grid">
          <div class="info-item">
            <div class="info-label">Sample Size</div>
            <div class="info-value">{{ dataset.ann_sample_size || dataset.meta_n_rows || 'Unknown' }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Year Updated</div>
            <div class="info-value">{{ dataset.ann_year_last_updated || 'Unknown' }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Years Data</div>
            <div class="info-value">{{ dataset.ann_years_data || 'Unknown' }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Country</div>
            <div class="info-value">{{ dataset.ann_country || 'Not specified' }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Sensitive Attributes Available</div>
            <div class="info-value">{{ dataset.ann_sensitive_attributes || 'Not specified' }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Default Scenario</div>
            <div class="info-value">{{ dataset.ann_default_scenario_sensitive_cols || 'Not specified' }}</div>
          </div>
        </div>
      </div>

      <div v-if="dataset.ann_description_public" class="dataset-section">
        <h2>Description</h2>
        <p class="description">{{ dataset.ann_description_public }}</p>
      </div>

      <div v-if="dataset.ann_notes_public" class="dataset-section">
        <h2>Notes</h2>
        <p class="description">{{ dataset.ann_notes_public }}</p>
      </div>

      <div class="dataset-section">
        <h2>Python Code Example</h2>
        <p>Use the following code to load this dataset with the fairml_datasets library:</p>
        <HighlightedCode
          :code="pythonExample"
          language="python"
          @copy="handleCodeCopy"
        />
      </div>

      <div class="dataset-section">
        <h2>CLI Example</h2>
        <p>You can also export this dataset using the command line:</p>
        <HighlightedCode
          :code="cliExample"
          language="bash"
          @copy="handleCodeCopy"
        />
      </div>

      <div v-if="dataset.ann_citation" class="dataset-section">
        <h2>Citation</h2>
        <HighlightedCode
          :code="dataset.ann_citation"
          language="bibtex"
          @copy="handleCodeCopy"
        />
      </div>


      <div v-if="relatedDatasets.length > 0" class="dataset-section">
        <h2>Related Datasets</h2>
        <ul class="related-datasets">
          <li v-for="related in relatedDatasets" :key="related.ann_new_dataset_id">
            <a :href="`/datasets/${related.ann_new_dataset_id}`">
              {{ related.ann_dataset_name }}
              <span v-if="related.ann_dataset_variant_description" class="variant-desc">
                ({{ related.ann_dataset_variant_description }})
              </span>
            </a>
          </li>
        </ul>
      </div>
    </div>

    <!-- Details Tab -->
    <div v-show="activeTab === 'details'" class="tab-content">
      <div class="dataset-section">
        <h2>Dataset Information</h2>
        <div class="info-grid">
          <div class="info-item">
            <div class="info-label">Dataset ID</div>
            <div class="info-value">{{ dataset.ann_new_dataset_id }}</div>
          </div>
          <div v-if="dataset.ann_full_dataset_id" class="info-item">
            <div class="info-label">Full Dataset ID</div>
            <div class="info-value">{{ dataset.ann_full_dataset_id }}</div>
          </div>
          <div v-if="dataset.ann_affiliation" class="info-item">
            <div class="info-label">Affiliation</div>
            <div class="info-value">{{ dataset.ann_affiliation }}</div>
          </div>
          <div v-if="dataset.ann_domain_freetext" class="info-item">
            <div class="info-label">Domain Details</div>
            <div class="info-value">{{ dataset.ann_domain_freetext }}</div>
          </div>
        </div>
      </div>

      <div class="dataset-section">
        <h2>Technical Details</h2>
        <div class="info-grid">
          <div class="info-item">
            <div class="info-label">Processing Required</div>
            <div class="info-value">{{ dataset.ann_processing || 'No' }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Format</div>
            <div class="info-value">{{ dataset.ann_format || 'Unknown' }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Is ZIP</div>
            <div class="info-value">{{ dataset.ann_is_zip ? 'Yes' : 'No' }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Is Large</div>
            <div class="info-value">{{ dataset.ann_is_large ? 'Yes' : 'No' }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Loading Status</div>
            <div class="info-value">{{ dataset.ann_loading_status || 'Unknown' }}</div>
          </div>
        </div>

        <div v-if="dataset.ann_filename_raw" class="info-grid mt-4">
          <div class="info-item">
            <div class="info-label">Raw Filename</div>
            <div class="info-value">{{ dataset.ann_filename_raw }}</div>
          </div>
        </div>
      </div>

      <div class="dataset-section">
        <h2>Dataset Statistics</h2>

        <h3>Dimensions</h3>
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-title">Number of Rows</div>
            <div class="metric-value">{{ dataset.meta_n_rows || 'N/A' }}</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Number of Columns</div>
            <div class="metric-value">{{ dataset.meta_n_cols || 'N/A' }}</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Number of Rows (Pre-transform)</div>
            <div class="metric-value">{{ dataset.meta_pretrans_n_rows || 'N/A' }}</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Number of Columns (Pre-transform)</div>
            <div class="metric-value">{{ dataset.meta_pretrans_n_cols || 'N/A' }}</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Unique Groups Count</div>
            <div class="metric-value">{{ dataset.meta_pretrans_unique_group_counts_pre_agg || 'N/A' }}</div>
            <div class="metric-desc">Number of unique groups before aggregation</div>
          </div>
        </div>

        <h3 class="mt-4">Data Types</h3>
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-title">Integer Columns</div>
            <div class="metric-value">{{ dataset.meta_prop_cols_int !== undefined ? dataset.meta_prop_cols_int : 'N/A' }}</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Float Columns</div>
            <div class="metric-value">{{ dataset.meta_prop_cols_float !== undefined ? dataset.meta_prop_cols_float : 'N/A' }}</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Boolean Columns</div>
            <div class="metric-value">{{ dataset.meta_prop_cols_bool !== undefined ? dataset.meta_prop_cols_bool : 'N/A' }}</div>
          </div>
        </div>

        <h3 class="mt-4">Feature Correlations</h3>
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-title">Avg. Feature Correlation</div>
            <div class="metric-value">{{ dataset.meta_average_absolute_correlation !== undefined ? dataset.meta_average_absolute_correlation : 'N/A' }}</div>
            <div class="metric-desc">Average absolute correlation between features</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Max Feature Correlation</div>
            <div class="metric-value">{{ dataset.meta_maximum_absolute_correlation !== undefined ? dataset.meta_maximum_absolute_correlation : 'N/A' }}</div>
            <div class="metric-desc">Maximum absolute correlation between features</div>
          </div>
        </div>

        <h3 class="mt-4">Missing Values</h3>
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-title">Rows with Missing Values</div>
            <div class="metric-value">{{ dataset.meta_pretrans_prop_NA_rows !== undefined ? dataset.meta_pretrans_prop_NA_rows : 'N/A' }}</div>
            <div class="metric-desc">Proportion of rows with at least one missing value</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Columns with Missing Values</div>
            <div class="metric-value">{{ dataset.meta_pretrans_prop_NA_cols !== undefined ? dataset.meta_pretrans_prop_NA_cols : 'N/A' }}</div>
            <div class="metric-desc">Proportion of columns with at least one missing value</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Cells with Missing Values</div>
            <div class="metric-value">{{ dataset.meta_pretrans_prop_NA_cells !== undefined ? dataset.meta_pretrans_prop_NA_cells : 'N/A' }}</div>
            <div class="metric-desc">Overall proportion of missing values</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Missing Values in Majority Group</div>
            <div class="metric-value">{{ dataset.meta_prop_NA_sens_majority !== undefined ? dataset.meta_prop_NA_sens_majority : 'N/A' }}</div>
            <div class="metric-desc">Proportion of missing values in the majority group</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Missing Values in Minority Group</div>
            <div class="metric-value">{{ dataset.meta_prop_NA_sens_minority !== undefined ? dataset.meta_prop_NA_sens_minority : 'N/A' }}</div>
            <div class="metric-desc">Proportion of missing values in the minority group</div>
          </div>
        </div>

        <div v-if="dataset.debug_meta_high_cardinality_strings" class="mt-4">
          <h3>High Cardinality Features</h3>
          <div class="info-item">
            <div class="info-label">High Cardinality String Columns</div>
            <div class="info-value">{{ dataset.debug_meta_high_cardinality_strings || 'None' }}</div>
          </div>
          <div class="info-item mt-4" v-if="dataset.debug_meta_high_cardinality_strings_and_counts">
            <div class="info-label">Column Counts</div>
            <div class="info-value">{{ dataset.debug_meta_high_cardinality_strings_and_counts }}</div>
          </div>
        </div>

        <div v-if="dataset.debug_meta_status" class="mt-4">
          <h3>Dataset Status</h3>
          <div class="info-item">
            <div class="info-label">Status</div>
            <div class="info-value">{{ dataset.debug_meta_status }}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Classification Task Tab -->
    <div v-show="activeTab === 'classification'" class="tab-content">
      <div class="dataset-section">
        <h2>Classification Task</h2>

        <div v-if="dataset.ann_dataset_variant_description" class="mb-4">
          <h3>Task Description</h3>
          <p>{{ dataset.ann_dataset_variant_description }}</p>
        </div>

        <div v-if="dataset.ann_typical_col_target" class="mb-4">
          <h3>Target Column</h3>
          <code>{{ dataset.ann_typical_col_target }}</code>
        </div>

        <div v-if="dataset.ann_target_lvl_good || dataset.ann_target_lvl_bad" class="task-levels">
          <h3>Target Levels</h3>
          <div class="levels-grid">
            <div v-if="dataset.ann_target_lvl_good" class="level good">
              <div class="level-label">Good/Positive Class</div>
              <div class="level-value">{{ dataset.ann_target_lvl_good }}</div>
            </div>
            <div v-if="dataset.ann_target_lvl_bad" class="level bad">
              <div class="level-label">Bad/Negative Class</div>
              <div class="level-value">{{ dataset.ann_target_lvl_bad }}</div>
            </div>
          </div>
        </div>
      </div>

      <div v-if="dataset.ann_typical_col_features && dataset.ann_typical_col_features !== '-'" class="dataset-section">
        <h3>Feature Columns</h3>
        <code>{{ dataset.ann_typical_col_features }}</code>
      </div>

      <div v-if="dataset.ann_colnames" class="dataset-section">
        <h3>All Columns</h3>
        <div class="columns-list">
          <code>{{ dataset.ann_colnames }}</code>
        </div>
      </div>
    </div>

    <!-- Fairness Tab -->
    <div v-show="activeTab === 'fairness'" class="tab-content">
      <div class="dataset-section">
        <h2>Fairness Information</h2>

        <div v-if="dataset.ann_sensitive_attributes" class="mb-4">
          <h3>Sensitive Attributes</h3>
          <p>{{ dataset.ann_sensitive_attributes }}</p>
        </div>

        <div v-if="dataset.ann_typical_col_sensitive" class="mb-4">
          <h3>Sensitive Columns Mapping</h3>
          <pre>{{ dataset.ann_typical_col_sensitive }}</pre>
        </div>

        <div class="mb-4">
          <h3>Fairness Metrics</h3>
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-title">Base Rate</div>
              <div class="metric-value">{{ dataset.meta_base_rate_target !== undefined ? dataset.meta_base_rate_target : 'N/A' }}</div>
              <div class="metric-desc">Overall positive outcome rate</div>
            </div>
            <div class="metric-card">
              <div class="metric-title">Base Rate Difference</div>
              <div class="metric-value">{{ dataset.meta_base_rate_difference }}</div>
              <div class="metric-desc">Difference between minority and majority group base rates</div>
            </div>
            <div class="metric-card">
              <div class="metric-title">Base Rate Ratio</div>
              <div class="metric-value">{{ dataset.meta_base_rate_ratio }}</div>
              <div class="metric-desc">Ratio of minority to majority group base rates</div>
            </div>
            <div class="metric-card">
              <div class="metric-title">Minority Base Rate</div>
              <div class="metric-value">{{ dataset.meta_base_rate_target_sens_minority }}</div>
              <div class="metric-desc">Positive outcome rate for minority group</div>
            </div>
            <div class="metric-card">
              <div class="metric-title">Majority Base Rate</div>
              <div class="metric-value">{{ dataset.meta_base_rate_target_sens_majority }}</div>
              <div class="metric-desc">Positive outcome rate for majority group</div>
            </div>
            <div class="metric-card">
              <div class="metric-title">Base Rate Gini Index</div>
              <div class="metric-value">{{ dataset.meta_base_rate_sens_gini !== undefined ? dataset.meta_base_rate_sens_gini : 'N/A' }}</div>
              <div class="metric-desc">Gini index for base rate distribution across sensitive groups</div>
            </div>
            <div class="metric-card">
              <div class="metric-title">Sensitive Attribute Predictability</div>
              <div class="metric-value">{{ dataset.meta_sens_predictability_roc_auc }}</div>
              <div class="metric-desc">ROC-AUC for predicting sensitive attribute from features (higher values indicate potential bias)</div>
            </div>
          </div>
        </div>

        <div class="mb-4">
          <h3>Sensitive Group Distribution</h3>
          <div class="metrics-grid cols-2">
            <div class="metric-card">
              <div class="metric-title">Minority Group Prevalence</div>
              <div class="metric-value">{{ dataset.meta_prev_sens_minority }}</div>
              <div class="metric-desc">Proportion of minority group in dataset</div>
            </div>
            <div class="metric-card">
              <div class="metric-title">Majority Group Prevalence</div>
              <div class="metric-value">{{ dataset.meta_prev_sens_majority }}</div>
              <div class="metric-desc">Proportion of majority group in dataset</div>
            </div>
            <div class="metric-card">
              <div class="metric-title">Prevalence Difference</div>
              <div class="metric-value">{{ dataset.meta_prev_sens_difference !== undefined ? dataset.meta_prev_sens_difference : 'N/A' }}</div>
              <div class="metric-desc">Difference between majority and minority group prevalence</div>
            </div>
            <div class="metric-card">
              <div class="metric-title">Prevalence Ratio</div>
              <div class="metric-value">{{ dataset.meta_prev_sens_ratio !== undefined ? dataset.meta_prev_sens_ratio : 'N/A' }}</div>
              <div class="metric-desc">Ratio of minority to majority group prevalence</div>
            </div>
            <div class="metric-card">
              <div class="metric-title">Sensitive Attribute Gini</div>
              <div class="metric-value">{{ dataset.meta_prev_sens_gini }}</div>
              <div class="metric-desc">Gini coefficient for sensitive attribute distribution (0 = perfect equality)</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Sources Tab -->
    <div v-show="activeTab === 'sources'" class="tab-content">
      <div class="dataset-section">
        <h2>Data Sources</h2>

        <div v-if="dataset.ann_main_url" class="mb-4">
          <h3>Main URL</h3>
          <a :href="dataset.ann_main_url" target="_blank" rel="noopener" class="source-link">
            {{ dataset.ann_main_url }}
            <span class="external-icon">↗</span>
          </a>
        </div>

        <div v-if="dataset.ann_download_url" class="mb-4">
          <h3>Download URL</h3>
          <a :href="dataset.ann_download_url" target="_blank" rel="noopener" class="source-link">
            {{ dataset.ann_download_url }}
            <span class="external-icon">↗</span>
          </a>
        </div>

        <div v-if="dataset.ann_related_urls" class="mb-4">
          <h3>Related URLs</h3>
          <p>{{ dataset.ann_related_urls }}</p>
        </div>

        <div v-if="dataset.ann_license" class="mb-4">
          <h3>License</h3>
          <p>{{ dataset.ann_license }}</p>
        </div>

        <div v-if="dataset.ann_custom_download" class="custom-download-notice">
          <h3>Custom Download Required</h3>
          <p>This dataset requires a custom download procedure.</p>
        </div>
      </div>
    </div>


  </div>
  <div v-else class="error-message">
    <h2>Dataset not found</h2>
    <p>The requested dataset could not be found. Please check the dataset ID and try again.</p>
    <a href="/" class="back-link">Return to dataset list</a>
  </div>
</template>

<style scoped>
:first-child > h2 {
  border-top: none;
  margin-top: 0;
  padding-top: 0;
}

.dataset-details {
  max-width: 1000px;
  margin: 0 auto 50px;
}

.dataset-header {
  margin-bottom: 24px;
}

.dataset-header h1 {
  margin-bottom: 8px;
}

.dataset-aliases {
  color: var(--vp-c-text-2);
  font-style: italic;
  margin-bottom: 16px;
}

.dataset-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}

.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 16px;
  font-size: 14px;
  font-weight: 500;
}

.badge.domain {
  background-color: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-dark);
}

.badge.data-type {
  background-color: var(--vp-c-gray-soft);
  color: var(--vp-c-gray-dark);
}

.badge.priority {
  background-color: var(--vp-c-green-soft);
  color: var(--vp-c-green-dark);
}

.badge.large {
  background-color: var(--vp-c-yellow-soft);
  color: var(--vp-c-yellow-dark);
}

.dataset-tabs {
  display: flex;
  border-bottom: 1px solid var(--vp-c-divider);
  margin-bottom: 24px;
  overflow-x: auto;
  white-space: nowrap;
}

.dataset-tabs button {
  padding: 12px 16px;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--vp-c-text-2);
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.dataset-tabs button:hover {
  color: var(--vp-c-text-1);
}

.dataset-tabs button.active {
  color: var(--vp-c-brand);
  border-bottom-color: var(--vp-c-brand);
}

.tab-content {
  animation: fadeIn 0.3s ease;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}

.mt-4 {
  margin-top: 16px;
}

.mb-4 {
  margin-bottom: 16px;
}

.info-item {
  padding: 12px;
  background-color: var(--vp-c-bg-soft);
  border-radius: 8px;
}

.info-label {
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}

.info-value {
  font-weight: 500;
}

.description {
  line-height: 1.6;
}

.levels-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 16px;
}

.level {
  padding: 12px;
  border-radius: 8px;
}

.level.good {
  background-color: var(--vp-c-green-soft);
}

.level.bad {
  background-color: var(--vp-c-red-soft);
}

.level-label {
  font-size: 0.9rem;
  font-weight: 500;
  margin-bottom: 4px;
}

.level-value {
  font-weight: 600;
}

.columns-list {
  max-height: 200px;
  overflow-y: auto;
}

.related-datasets {
  list-style-type: none;
  padding: 0;
}

.related-datasets li {
  margin-bottom: 8px;
}

.related-datasets a {
  color: var(--vp-c-brand);
  text-decoration: none;
}

.related-datasets a:hover {
  text-decoration: underline;
}

.variant-desc {
  font-size: 0.9em;
  color: var(--vp-c-text-2);
}

.source-link {
  display: inline-flex;
  align-items: center;
  color: var(--vp-c-brand);
  text-decoration: none;
  word-break: break-all;
}

.source-link:hover {
  text-decoration: underline;
}

.external-icon {
  margin-left: 4px;
  font-size: 0.8em;
}

.custom-download-notice {
  padding: 16px;
  background-color: var(--vp-c-warning-soft);
  border-radius: 8px;
}

.error-message {
  text-align: center;
  padding: 48px 24px;
  background-color: var(--vp-c-bg-soft);
  border-radius: 8px;
  margin: 32px 0;
}

.back-link {
  display: inline-block;
  margin-top: 16px;
  color: var(--vp-c-brand);
  text-decoration: none;
}

.back-link:hover {
  text-decoration: underline;
}

.code-block-wrapper {
  position: relative;
  margin-top: 16px;
  border-radius: 8px;
  overflow: hidden;
}

.loading-code {
  background-color: var(--vp-code-block-bg, #1e1e1e);
  color: var(--vp-c-text-2);
  padding: 20px;
  border-radius: 8px;
  text-align: center;
}

:deep(.shiki) {
  background-color: transparent !important;
}

.copy-button {
  position: absolute;
  top: 8px;
  right: 8px;
  padding: 4px 8px;
  font-size: 0.8rem;
  background-color: rgba(30, 30, 30, 0.5);
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  cursor: pointer;
  color: var(--vp-c-text-2);
  transition: all 0.2s;
  z-index: 2;
}

.copy-button:hover {
  background-color: rgba(60, 60, 60, 0.7);
  color: var(--vp-c-text-1);
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
  margin-top: 16px;
}

.metrics-grid.cols-2 {
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
}

.metric-card {
  background-color: var(--vp-c-bg-soft);
  border-radius: 8px;
  padding: 16px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
}

.metric-title {
  font-weight: 600;
  margin-bottom: 8px;
  color: var(--vp-c-text-1);
}

.metric-value {
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--vp-c-brand);
  margin-bottom: 4px;
}

.metric-desc {
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
  line-height: 1.4;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@media (max-width: 768px) {
  .dataset-tabs {
    margin: 0 -16px 24px;
    padding: 0 16px;
  }

  .dataset-tabs button {
    padding: 12px 10px;
    font-size: 0.9rem;
  }

  .info-grid {
    grid-template-columns: 1fr;
  }
}
.dataset-warning {
  padding: 16px;
  background-color: var(--vp-c-warning-soft);
  border-radius: 8px;
  color: var(--vp-c-warning-dark, #b26a00);
  margin-bottom: 16px;
  font-size: 1rem;
  display: flex;
  align-items: center;
  gap: 8px;
}

</style>
