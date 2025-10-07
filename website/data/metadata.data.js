import fs from "node:fs";
import path from "node:path";

export default {
  watch: ["./metadata.json"],

  load(watchedFiles) {
    // If we have watched files (in development), use the first one
    if (watchedFiles && watchedFiles.length > 0) {
      const content = fs.readFileSync(watchedFiles[0], "utf-8");
      return processMetadata(JSON.parse(content));
    }

    // Otherwise (in production), load the metadata file directly
    const metadataPath = path.resolve(__dirname, "../metadata.json");
    const content = fs.readFileSync(metadataPath, "utf-8");
    return processMetadata(JSON.parse(content));
  },
};

// Helper function to format decimal values
function processMetadata(metadata) {
  return metadata.map((item) => {
    // Create a copy of the item to avoid mutating the original
    const formattedItem = { ...item };

    // Format percentage values with 2 decimal places
    const percentageFields = [
      "meta_prev_sens_minority",
      "meta_prev_sens_majority",
      "meta_prev_sens_difference",
      "meta_base_rate_target",
      "meta_base_rate_target_sens_minority",
      "meta_base_rate_target_sens_majority",
      "meta_base_rate_difference",
      "meta_prop_cols_int",
      "meta_prop_cols_float",
      "meta_prop_cols_bool",
      "meta_prop_NA_sens_majority",
      "meta_prop_NA_sens_minority",
    ];

    // Format decimal values with 2 decimal places
    const decimalFields = [
      "meta_base_rate_ratio",
      "meta_prev_sens_ratio",
      "meta_prev_sens_gini",
      "meta_base_rate_sens_gini",
      "meta_sens_predictability_roc_auc",
      "meta_average_absolute_correlation",
      "meta_maximum_absolute_correlation",
    ];

    // Apply formatting to percentage fields
    percentageFields.forEach((field) => {
      if (formattedItem[field] !== undefined && formattedItem[field] !== null) {
        formattedItem[field] = Number(formattedItem[field].toFixed(4));
      }
    });

    // Apply formatting to decimal fields
    decimalFields.forEach((field) => {
      if (formattedItem[field] !== undefined && formattedItem[field] !== null) {
        formattedItem[field] = Number(formattedItem[field].toFixed(2));
      }
    });

    return formattedItem;
  });
}
