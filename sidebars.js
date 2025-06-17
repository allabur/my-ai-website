/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a "Next" and "Previous" button at the bottom of each doc
 - provide a way to navigate between docs

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Acerca de mí',
      items: [
        'about_me/intro',
        'about_me/experiencia',
        'about_me/educacion',
      ],
    },
    {
      type: 'category',
      label: 'Mis Proyectos',
      items: [
        'proyectos/proyecto-ia',
        'proyectos/aplicaciones-web',
        'proyectos/investigacion',
      ],
    },
    {
      type: 'category',
      label: 'Herramientas',
      items: [
        'herramientas/python',
        'herramientas/javascript',
        'herramientas/ia-ml',      ],
    },
    {
      type: 'category',
      label: 'Tutoriales',
      items: [
        'tutoriales/programacion',
        'tutoriales/ia',
        'tutoriales/desarrollo-web',
      ],
    },
  ],
};

module.exports = sidebars;
