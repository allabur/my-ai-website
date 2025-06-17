# Mi Sitio Web Personal

Este es mi sitio web personal construido con [Docusaurus 2](https://docusaurus.io/), un generador de sitios web estáticos moderno.

## Instalación

```bash
npm install
```

## Desarrollo Local

```bash
npm start
```

Este comando inicia un servidor de desarrollo local y abre una ventana del navegador. La mayoría de los cambios se reflejan en tiempo real sin necesidad de reiniciar el servidor.

## Construcción

```bash
npm run build
```

Este comando genera contenido estático en el directorio `build` y puede ser servido usando cualquier servicio de alojamiento de contenido estático.

## Despliegue

### Usando SSH:

```bash
USE_SSH=true npm run deploy
```

### Sin usar SSH:

```bash
GIT_USER=<Tu nombre de usuario de GitHub> npm run deploy
```

Si estás usando GitHub Pages para el alojamiento, este comando es una forma conveniente de construir el sitio web y subirlo a la rama `gh-pages`.

## Personalización

- Edita `docusaurus.config.js` para configurar el sitio
- Modifica los archivos en `docs/` para añadir contenido
- Personaliza los estilos en `src/css/custom.css`
- Añade imágenes en `static/img/`

## Estructura del Proyecto

```
my-ai-website/
├── blog/                    # Posts del blog
├── docs/                    # Documentos principales
├── src/
│   ├── components/          # Componentes React personalizados
│   ├── css/                 # Estilos personalizados
│   └── pages/               # Páginas personalizadas
├── static/                  # Archivos estáticos
│   └── img/                 # Imágenes
├── docusaurus.config.js     # Configuración principal
├── package.json             # Dependencias
└── sidebars.js             # Configuración de barras laterales
```
