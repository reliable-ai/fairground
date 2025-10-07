<script setup>
import { AgGridVue } from "ag-grid-vue3";
// import "ag-grid-community/styles/ag-grid.css";
// import "ag-grid-community/styles/ag-theme-alpine.css";
import { computed, onMounted, ref } from "vue";
import { data as metadata } from "../data/metadata.data.js";

import { AllCommunityModule, ModuleRegistry } from "ag-grid-community";

ModuleRegistry.registerModules([AllCommunityModule]);

const props = defineProps({
  fullscreen: {
    type: Boolean,
    default: false,
  },
});

const tableElement = ref(null);
const gridApi = ref(null);

// AG Grid column definitions
const columnDefs = computed(() => [
  {
    headerName: "Name",
    field: "name",
    sortable: true,
    filter: props.fullscreen,
    minWidth: 200,
    pinned: "left",
    valueFormatter: (params) => params.value?.toString() || "N/A",
    cellRenderer: (params) => {
      return `<a href="/datasets/${params.data.id}" class="dataset-name-link">
                <strong>${params.value}</strong>
              </a>`;
    },
  },
  {
    headerName: "ID",
    field: "id",
    sortable: true,
    filter: props.fullscreen,
    minWidth: 120,
    cellRenderer: (params) => {
      return `<div class="id-container">
                <code>${params.value}</code>
                <button class="copy-button" data-clipboard-text="${params.value}" title="Copy to clipboard">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1-2-2h9a2 2 0 0 1-2 2v1"></path>
                  </svg>
                </button>
              </div>`;
    },
    onCellClicked: (params) => {
      // Only trigger if the click was on the copy button
      const target = params.event.target;
      if (target.closest && target.closest(".copy-button")) {
        params.event.stopPropagation(); // Prevent row selection
        const text = params.value;
        navigator.clipboard
          .writeText(text)
          .then(() => {
            const button = target.closest(".copy-button");
            button.classList.add("copied");
            setTimeout(() => button.classList.remove("copied"), 1000);
          })
          .catch((err) => console.error("Failed to copy text: ", err));
      }
    },
  },
  {
    headerName: "Rows",
    field: "n_rows",
    sortable: true,
    filter: props.fullscreen ? "agNumberColumnFilter" : false,
    minWidth: 120,
    type: "numericColumn",
    valueFormatter: (params) => {
      const value = params.value;
      if (value === null || value === undefined) return "N/A";
      return value.toLocaleString();
    },
    cellRenderer: (params) => {
      const value = params.value;
      if (value === null || value === undefined) return "N/A";
      return value.toLocaleString();
    },
  },
  {
    headerName: "Cols",
    field: "n_cols",
    sortable: true,
    filter: props.fullscreen ? "agNumberColumnFilter" : false,
    minWidth: 90,
    type: "numericColumn",
    valueFormatter: (params) => {
      const value = params.value;
      if (value === null || value === undefined) return "N/A";
      return value.toLocaleString();
    },
    cellRenderer: (params) => {
      const value = params.value;
      if (value === null || value === undefined) return "N/A";
      return value.toLocaleString();
    },
  },
  {
    headerName: "Domain",
    field: "domain",
    sortable: true,
    filter: props.fullscreen ? "agSetColumnFilter" : false,
    minWidth: 150,
    valueFormatter: (params) => params.value?.toString() || "N/A",
  },
  {
    headerName: "Sensitive Attributes",
    field: "sensitive_attrs_array",
    sortable: false,
    minWidth: 250,
    filter: props.fullscreen ? "agSetColumnFilter" : false,
    valueFormatter: (params) => {
      const values = params.value || [];
      return Array.isArray(values) ? values.join(", ") : "N/A";
    },
    cellRenderer: (params) => {
      const values = params.value || [];
      return values
        .map((value) => {
          return `<span class="attr-tag">${value}</span>`;
        })
        .join(" ");
    },
  },
  {
    headerName: "Year",
    field: "year",
    sortable: true,
    filter: props.fullscreen ? "agSetColumnFilter" : false,
    minWidth: 100,
    valueFormatter: (params) => params.value?.toString() || "N/A",
  },
  {
    headerName: "Country",
    field: "country",
    sortable: true,
    filter: props.fullscreen ? "agSetColumnFilter" : false,
    minWidth: 120,
    valueFormatter: (params) => params.value?.toString() || "N/A",
  },
  {
    headerName: "License",
    field: "license",
    sortable: true,
    filter: props.fullscreen ? "agSetColumnFilter" : false,
    minWidth: 150,
    valueFormatter: (params) => params.value?.toString() || "Unknown",
    cellRenderer: (params) => {
      const value = params.value || "Unknown";
      if (value === "?" || value === "null" || value === null) {
        return "<span class='license-unknown'>Unknown</span>";
      }
      return value;
    },
  },
  {
    headerName: "Base Rate",
    field: "base_rate",
    sortable: true,
    filter: props.fullscreen ? "agNumberColumnFilter" : false,
    minWidth: 100,
    type: "numericColumn",
    valueFormatter: (params) => {
      const value = params.value;
      if (value === null || value === undefined) return "N/A";
      return `${(value * 100).toFixed(1)}%`;
    },
    cellRenderer: (params) => {
      const value = params.value;
      if (value === null || value === undefined) return "N/A";
      return `${(value * 100).toFixed(1)}%`;
    },
  },
  {
    headerName: "Sens. Minority/Majority",
    field: "sens_minority",
    sortable: true,
    filter: props.fullscreen ? "agNumberColumnFilter" : false,
    minWidth: 180,
    type: "numericColumn",
    valueFormatter: (params) => {
      const value = params.value;
      const majorityValue = params.data.sens_majority;
      if (
        value === null ||
        value === undefined ||
        majorityValue === null ||
        majorityValue === undefined
      )
        return "N/A";
      return `${(value * 100).toFixed(1)}% / ${(majorityValue * 100).toFixed(1)}%`;
    },
    cellRenderer: (params) => {
      const value = params.value;
      const majorityValue = params.data.sens_majority;
      if (
        value === null ||
        value === undefined ||
        majorityValue === null ||
        majorityValue === undefined
      )
        return "N/A";
      return `${(value * 100).toFixed(1)}% / ${(majorityValue * 100).toFixed(1)}%`;
    },
  },
  {
    headerName: "Base Rate Diff.",
    field: "base_rate_diff",
    sortable: true,
    filter: props.fullscreen ? "agNumberColumnFilter" : false,
    minWidth: 140,
    type: "numericColumn",
    valueFormatter: (params) => {
      const value = params.value;
      if (value === null || value === undefined) return "N/A";
      return value.toString();
    },
  },
  {
    headerName: "Base Rate Ratio",
    field: "base_rate_ratio",
    sortable: true,
    filter: props.fullscreen ? "agNumberColumnFilter" : false,
    minWidth: 140,
    type: "numericColumn",
    valueFormatter: (params) => {
      const value = params.value;
      if (value === null || value === undefined) return "N/A";
      return value.toString();
    },
  },
]);

// Prepare data for AG Grid
const rowData = ref([]);

// Grid options
const gridOptions = ref({
  defaultColDef: {
    resizable: true,
    sortable: true,
    valueFormatter: (params) => {
      if (params.value === null || params.value === undefined) return "N/A";
      if (typeof params.value === "object") return JSON.stringify(params.value);
      return params.value.toString();
    },
  },
  rowHeight: 48,
  headerHeight: 48,
  animateRows: true,
  suppressCellFocus: true,
});

// Event when the grid is ready
const onGridReady = (params) => {
  gridApi.value = params.api;

  // Size columns to fit
  if (gridApi.value) {
    gridApi.value.sizeColumnsToFit();
  }
};

onMounted(() => {
  try {
    // Prepare data for AG Grid
    rowData.value = metadata.map((item) => ({
      id: item.ann_new_dataset_id || item.id,
      name: item.ann_dataset_name,
      base_id: item.ann_dataset_base_id,
      aliases: item.ann_dataset_aliases,
      domain: item.ann_domain_class,
      domain_details: item.ann_domain_freetext,
      samples: item.ann_sample_size,
      sensitive_attrs: item.ann_sensitive_attributes,
      affiliation: item.ann_affiliation,
      year: item.ann_year_last_updated,
      years_data: item.ann_years_data,
      country: item.ann_country,
      license: item.ann_license,
      format: item.ann_format,
      variant: item.ann_dataset_variant_description,
      // Parse sensitive attributes into array (handling different formats)
      sensitive_attrs_array: Array.isArray(item.ann_sensitive_attributes)
        ? item.ann_sensitive_attributes
        : (item.ann_sensitive_attributes || "")
            .replace("?", "")
            .split(/[,;]/)
            .map((attr) => attr.trim())
            .filter((attr) => attr !== ""),
      // Calculate base rate difference for fairness metrics
      base_rate_diff: item.meta_base_rate_difference,
      base_rate_ratio: item.meta_base_rate_ratio,
      // Additional meta information
      n_rows: item.meta_n_rows,
      n_cols: item.meta_n_cols,
      base_rate: item.meta_base_rate_target,
      sens_minority: item.meta_prev_sens_minority,
      sens_majority: item.meta_prev_sens_majority,
      base_rate_minority: item.meta_base_rate_target_sens_minority,
      base_rate_majority: item.meta_base_rate_target_sens_majority,
      sens_predictability: item.meta_sens_predictability_roc_auc,
    }));
  } catch (error) {
    console.error("Error setting up dataset table:", error);
    if (tableElement.value) {
      tableElement.value.innerHTML =
        "Error loading datasets. Please try again later.";
    }
  }
});
</script>

<template>
  <div class="metadata-table-container" :class="{ 'fullscreen': fullscreen }">
    <AgGridVue
      ref="tableElement"
      class="metadata-table"
      :columnDefs="columnDefs"
      :rowData="rowData"
      :gridOptions="gridOptions"
      @grid-ready="onGridReady"
    />
  </div>
</template>

<style>
.metadata-table-container {
  width: 100%;
  height: 500px;
  display: flex;
}

.metadata-table-container.fullscreen {
  height: calc(100vh - 80px);
}

.metadata-table {
  width: 100%;
  flex: 1;
}

.copy-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
  color: var(--vp-c-text-2);
  cursor: pointer;
  padding: 2px;
  border-radius: 4px;
  opacity: 0.7;
  transition: all 0.2s ease;
}

.copy-button:hover {
  opacity: 1;
  color: var(--vp-c-brand);
  background-color: var(--vp-c-bg-soft);
}

.copy-button.copied {
  color: var(--vp-c-brand);
  background-color: var(--vp-c-brand-soft);
}

/* Improve responsive handling for ID column */
@media (max-width: 768px) {
  .id-container {
    gap: 4px;
  }

  .id-container code {
    font-size: 0.85em;
  }

  .copy-button svg {
    width: 12px;
    height: 12px;
  }
}

/* Dataset name link styling */
a.dataset-name-link {
  color: var(--vp-c-text-1);
  text-decoration: none;
  transition: color 0.2s ease;
  display: inline-block;
  padding: 2px 0;
}

.dataset-name-link:hover {
  color: var(--vp-c-brand);
  text-decoration: underline;
}

.dataset-name-link strong {
  font-weight: 600;
}

/* Sensitive attribute tags styling */
.attr-tag {
  display: inline-block;
  background-color: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-dark);
  padding: 2px 8px;
  margin: 2px;
  border-radius: 12px;
  white-space: nowrap;
  line-height: 1;
}

.license-unknown {
  font-style: italic;
}
</style>
