import { defineConfig } from "vitepress";

interface SidebarItem {
  text: string;
  link?: string;
  items?: SidebarItem[];
  collapsed?: boolean;
  target?: string;
}

const sidebar: SidebarItem[] = [
  {
    text: "Overview",
    items: [
      { text: "What is FairGround?", link: "/about" },
      { text: "View All Datasets", link: "/fullscreen", target: "_blank" },
    ],
  },
  {
    text: "Collections",
    items: [
      {
        text: "DeCorrelated",
        collapsed: true,
        items: [
          {
            link: "collections/DeCorrelatedSmall",
            text: "DeCorrelated (small)",
          },
          {
            link: "collections/DecorrelatedLarge",
            text: "DeCorrelated (large)",
          },
        ],
      },

      {
        text: "Permissively Licensed",
        collapsed: true,
        items: [
          {
            link: "collections/PermissivelyLicensedSmall",
            text: "Perm. Licensed (small)",
          },
          {
            link: "collections/PermissivelyLicensedLarge",
            text: "Perm. Licensed (large)",
          },
          {
            link: "collections/PermissivelyLicensedFull",
            text: "Perm. Licensed (full)",
          },
        ],
      },
      {
        text: "Geographically Diverse",
        collapsed: true,
        items: [
          {
            link: "collections/GeographicSmall",
            text: "Geogr. Diverse (small)",
          },
          {
            link: "collections/GeographicLarge",
            text: "Geogr. Diverse (large)",
          },
          {
            link: "collections/GeographicFull",
            text: "Geogr. Diverse (full)",
          },
        ],
      },
    ],
  },
];

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "FairGround",
  description: "A framework and corpus of datasets for fairness research.",
  base: "/fairground/",
  head: [
    ['link', { rel: 'icon', type: 'image/png', href: '/fairground/favicon.png' }]
  ],
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: "Home", link: "/" },
      { text: "About", link: "/about" },
      { text: "Datasets", link: "/fullscreen" },
      {
        text: "Package",
        link: "/docs/",
        target: "_blank",
      },
    ],

    sidebar,

    socialLinks: [
      {
        icon: "github",
        link: "https://github.com/reliable-ai/fairground",
      },
    ],

    footer: {
      message: 'Released under <a href="https://github.com/reliable-ai/fairground?tab=readme-ov-file#license">CC-BY-4.0 and GPL-3.0</a>.',
      copyright: 'Copyright © 2025-present FairGround Authors'
    },

    search: {
      provider: "local",
      options: {
        detailedView: false,
      },
    },
  },
});
