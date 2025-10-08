---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "FairGround"
  text: "A corpus of datasets for fairness research."
  tagline: Explore datasets commonly used in algorithmic fairness research
  image:
    src: /logo.svg
    alt: logo
  actions:
    - theme: brand
      text: Browse Datasets
      link: /fullscreen
    - theme: alt
      text: About This Project
      link: /about

features:
  - title: Comprehensive Collection
    details: Access a wide range of datasets used in fairness research across various domains.
  - title: Detailed Documentation
    details: Each dataset includes detailed metadata about sensitive attributes, its classification task, data sources, and more.
  - title: Easy Exploration
    details: Filter, sort, and search datasets to find exactly what you need for your research.
---

<div class="datasets-container" id="datasets">
  <MetadataTable />
</div>

<script setup>
import MetadataTable from './components/MetadataTable.vue';
</script>

<style>
.datasets-container {
  margin: 3rem auto;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  #datasets-table {
    font-size: 0.9rem;
  }
  
  .tabulator-header .tabulator-col,
  .tabulator-row .tabulator-cell {
    padding: 8px 6px;
  }
  
  .tabulator-page {
    padding: 3px 8px;
  }
}
</style>

Built with support from

![BERD@NFDI](logo_berd.png)
