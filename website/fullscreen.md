---
layout: page
title: Fairness Datasets - Fullscreen View
sidebar: false
---

<div class="vp-doc fullscreen-page">
  <MetadataTable :fullscreen="true" />
</div>

<script setup>
import MetadataTable from './components/MetadataTable.vue';
</script>

<style>
.fullscreen-page {
  display: flex;
  flex-direction: column;
  padding: 0;
  margin: 0;
  max-width: none;
}

.VPDocFooter {
  display: none !important;
}
</style>
