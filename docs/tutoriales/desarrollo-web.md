# Desarrollo Web

Guía completa para el desarrollo web moderno, desde frontend hasta backend.

## 🌐 Introducción al Desarrollo Web

### ¿Qué es el Desarrollo Web?

El desarrollo web es el proceso de crear aplicaciones y sitios web que funcionan en internet.

### Tecnologías Fundamentales

1. **Frontend (Cliente)**
   - HTML: Estructura
   - CSS: Diseño y estilo
   - JavaScript: Interactividad

2. **Backend (Servidor)**
   - Lenguajes: Python, Node.js, PHP, Java
   - Bases de datos: MySQL, PostgreSQL, MongoDB
   - APIs: REST, GraphQL

## 🎨 Frontend Development

### HTML5

```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mi Web</title>
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href="#inicio">Inicio</a></li>
                <li><a href="#about">Acerca</a></li>
                <li><a href="#contacto">Contacto</a></li>
            </ul>
        </nav>
    </header>
    <main>
        <section id="inicio">
            <h1>Bienvenido</h1>
            <p>Esta es mi página web.</p>
        </section>
    </main>
</body>
</html>
```

### CSS3 Moderno

```css
/* Variables CSS */
:root {
    --primary-color: #007bff;
    --secondary-color: #6c757d;
    --font-family: 'Inter', sans-serif;
}

/* Flexbox Layout */
.container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
}

/* Grid Layout */
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
}

/* Responsive Design */
@media (max-width: 768px) {
    .container {
        flex-direction: column;
        padding: 1rem;
    }
}
```

### JavaScript ES6+

```javascript
// Arrow Functions
const fetchData = async (url) => {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
    }
};

// Destructuring
const { name, email } = user;

// Template Literals
const message = `Hola ${name}, tu email es ${email}`;

// Modules
export { fetchData };
import { fetchData } from './utils.js';
```

## 🚀 Frameworks y Librerías

### Frontend Frameworks

1. **React**
   - Componentes reutilizables
   - Virtual DOM
   - Ecosistema robusto

2. **Vue.js**
   - Curva de aprendizaje suave
   - Reactividad
   - Flexibilidad

3. **Angular**
   - Framework completo
   - TypeScript
   - Inyección de dependencias

### CSS Frameworks

- **Bootstrap**: Componentes predefinidos
- **Tailwind CSS**: Utility-first
- **Bulma**: Flexbox-based

## 🔧 Backend Development

### Node.js con Express

```javascript
const express = require('express');
const app = express();

// Middleware
app.use(express.json());
app.use(express.static('public'));

// Rutas
app.get('/', (req, res) => {
    res.json({ message: 'Hola Mundo!' });
});

app.post('/api/users', (req, res) => {
    const { name, email } = req.body;
    // Lógica para crear usuario
    res.status(201).json({ id: 1, name, email });
});

app.listen(3000, () => {
    console.log('Servidor en puerto 3000');
});
```

### Bases de Datos

#### MongoDB con Mongoose

```javascript
const mongoose = require('mongoose');

// Schema
const userSchema = new mongoose.Schema({
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    createdAt: { type: Date, default: Date.now }
});

// Model
const User = mongoose.model('User', userSchema);

// Operaciones CRUD
const createUser = async (userData) => {
    const user = new User(userData);
    return await user.save();
};
```

## 🛠️ Herramientas de Desarrollo

### Control de Versiones
- Git y GitHub
- Branching strategies
- Pull requests

### Bundlers y Build Tools
- **Webpack**: Module bundler
- **Vite**: Fast build tool
- **Parcel**: Zero-config bundler

### Testing
- **Jest**: JavaScript testing
- **Cypress**: E2E testing
- **Postman**: API testing

## 🌟 Mejores Prácticas

### Performance
1. **Optimización de imágenes**
2. **Lazy loading**
3. **Code splitting**
4. **CDN usage**

### Seguridad
1. **HTTPS obligatorio**
2. **Validación de inputs**
3. **Autenticación JWT**
4. **Sanitización de datos**

### SEO
1. **Meta tags**
2. **Estructura semántica**
3. **URLs limpias**
4. **Schema markup**

## 🎯 Proyecto Full Stack

### E-commerce Básico

**Frontend**: React + TypeScript
**Backend**: Node.js + Express
**Database**: MongoDB
**Auth**: JWT
**Payments**: Stripe

## 📚 Recursos de Aprendizaje

- **MDN Web Docs**: Documentación oficial
- **FreeCodeCamp**: Cursos gratuitos
- **The Odin Project**: Curriculum completo
- **JavaScript.info**: Guía de JavaScript

---

*Última actualización: Junio 2025*
