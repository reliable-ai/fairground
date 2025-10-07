const fs = require("node:fs");
const path = require("node:path");

// Read the metadata.json file
const metadata = JSON.parse(fs.readFileSync("./data/metadata.json", "utf8"));

// Create the datasets directory if it doesn't exist
const datasetsDir = path.join(".", "datasets");
if (!fs.existsSync(datasetsDir)) {
  fs.mkdirSync(datasetsDir, { recursive: true });
}

// Generate a markdown page for each dataset
metadata.forEach((dataset) => {
  const datasetId = dataset.ann_new_dataset_id;
  const datasetName = dataset.ann_dataset_name;

  const content = `---
title: ${datasetName}
---

# ${datasetName}

<DatasetDetails id="${datasetId}" />
`;

  fs.writeFileSync(path.join(datasetsDir, `${datasetId}.md`), content);
  console.log(`Generated page for ${datasetId}`);
});

console.log("All pages generated successfully!");
