// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import { themes as prismThemes } from "prism-react-renderer";

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "Tu Nombre - Portfolio Personal",
  tagline: "Desarrollador de IA | Ingeniero | Investigador",
  favicon: "img/favicon.ico",
  // Set the production url of your site here
  url: "https://localhost:3000",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "github", // Usually your GitHub org/user name.
  projectName: "my-ai-website", // Usually your repo name.
  deploymentBranch: "gh-pages",
  trailingSlash: false,

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to set htmlLang to 'zh-Hans'.
  i18n: {
    defaultLocale: "es",
    locales: ["es", "en"],
  },

  markdown: {
    mermaid: true,
  },
  themes: ["@docusaurus/theme-mermaid"],

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: "./sidebars.js",
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl: "https://github.com/github/my-ai-website/tree/main/",
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
        },
        blog: {
          showReadingTime: true,
          readingTime: ({ content, frontMatter, defaultReadingTime, locale }) =>
            defaultReadingTime({
              content,
              locale,
              options: { wordsPerMinute: 300 },
            }),
          feedOptions: {
            type: ["rss", "atom"],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl: "https://github.com/github/my-ai-website/tree/main/",
          // Useful options to enforce blogging best practices
          onInlineTags: "warn",
          onInlineAuthors: "warn",
          onUntruncatedBlogPosts: "warn",
        },
        theme: {
          customCss: "./src/css/custom.css",
        },
        gtag: {
          trackingID: "G-XXXXXXXXXX", // Replace with your Google Analytics ID
          anonymizeIP: true,
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: "img/docusaurus-social-card.jpg",
      metadata: [
        {
          name: "keywords",
          content:
            "inteligencia artificial, desarrollo web, investigación, IA, machine learning",
        },
        { name: "twitter:card", content: "summary_large_image" },
      ],
      navbar: {
        title: "Tu Nombre",
        logo: {
          alt: "Mi Logo",
          src: "img/logo.svg",
        },
        items: [
          {
            type: "docSidebar",
            sidebarId: "tutorialSidebar",
            position: "left",
            label: "Acerca de mí",
          },
          { to: "/blog", label: "Blog", position: "left" },
          {
            type: "localeDropdown",
            position: "right",
          },
          {
            href: "https://github.com",
            label: "GitHub",
            position: "right",
          },
          {
            href: "https://linkedin.com",
            label: "LinkedIn",
            position: "right",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Acerca de mí",
            items: [
              {
                label: "Mi trabajo",
                to: "/docs/about_me/intro",
              },
              {
                label: "Proyectos",
                to: "/docs/category/mis-proyectos",
              },
            ],
          },
          {
            title: "Comunidad",
            items: [
              {
                label: "LinkedIn",
                href: "https://linkedin.com",
              },
              {
                label: "Twitter",
                href: "https://twitter.com",
              },
              {
                label: "YouTube",
                href: "https://youtube.com",
              },
            ],
          },
          {
            title: "Más",
            items: [
              {
                label: "Blog",
                to: "/blog",
              },
              {
                label: "GitHub",
                href: "https://github.com",
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Tu Nombre. Construido con Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: [
          "python",
          "javascript",
          "typescript",
          "bash",
          "json",
          "yaml",
        ],
      },
      algolia: {
        // The application ID provided by Algolia
        appId: "YOUR_APP_ID",
        // Public API key: it is safe to commit it
        apiKey: "YOUR_SEARCH_API_KEY",
        indexName: "YOUR_INDEX_NAME",
        // Optional: see doc section below
        contextualSearch: true,
        // Optional: Specify domains where the navigation should occur through window.location instead on history.push. Useful when our Algolia config crawls multiple documentation sites and we want to navigate with window.location.href to them.
        externalUrlRegex: "external\\.com|domain\\.com",
        // Optional: Replace parts of the item URLs from Algolia. Useful when using the same search index for multiple deployments using a different baseUrl. You can use regexp or string in the `from` param. For example: localhost:3000 vs myCompany.github.io/myProject/
        replaceSearchResultPathname: {
          from: "/docs/", // or as a regexp: /\/docs\//
          to: "/",
        },
        // Optional: Algolia search parameters
        searchParameters: {},
        // Optional: path for search page that enabled by default (`false` to disable it)
        searchPagePath: "search",
      },
    }),
};

export default config;
