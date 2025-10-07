import Theme from "vitepress/theme";
// https://vitepress.dev/guide/custom-theme
import { h } from "vue";
import "./style.css";

import CollectionDetails from "../../components/CollectionDetails.vue";
// @ts-ignore
import DatasetDetails from "../../components/DatasetDetails.vue";

export default {
  extends: Theme,
  Layout: () => {
    return h(Theme.Layout, null, {
      // https://vitepress.dev/guide/extending-default-theme#layout-slots
    });
  },
  enhanceApp({ app, router, siteData }) {
    app.component("DatasetDetails", DatasetDetails);
    app.component("CollectionDetails", CollectionDetails);
  },
};
