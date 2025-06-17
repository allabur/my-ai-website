---
sidebar_position: 1
---

# JavaScript Moderno para Desarrollo Web

JavaScript ha evolucionado enormemente en los últimos años. En esta guía exploraremos las características y herramientas más importantes para el desarrollo web moderno.

## 🚀 JavaScript ES2024+ Características

### Nuevas Características del Lenguaje

#### 1. Optional Chaining y Nullish Coalescing

```javascript
// Optional Chaining (?.)
const user = {
  profile: {
    social: {
      twitter: '@usuario'
    }
  }
};

// Forma segura de acceder a propiedades anidadas
const twitterHandle = user?.profile?.social?.twitter;
const instagramHandle = user?.profile?.social?.instagram ?? 'No disponible';

// Con arrays
const firstPost = user?.posts?.[0]?.title;
```

#### 2. Destructuring Avanzado

```javascript
// Destructuring con valores por defecto
const { name = 'Anónimo', age = 0, city = 'No especificada' } = user;

// Rest/Spread con objetos
const { password, ...publicUserData } = user;
const updatedUser = { ...user, lastLogin: new Date() };

// Destructuring en parámetros de función
function createUser({ name, email, role = 'user' }) {
  return { name, email, role, id: generateId() };
}
```

#### 3. Async/Await Moderno

```javascript
// Manejo de errores con async/await
async function fetchUserData(userId) {
  try {
    const response = await fetch(`/api/users/${userId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const userData = await response.json();
    return userData;
  } catch (error) {
    console.error('Error fetching user data:', error);
    throw error;
  }
}

// Promesas en paralelo
async function loadDashboardData() {
  try {
    const [users, posts, analytics] = await Promise.all([
      fetchUsers(),
      fetchPosts(),
      fetchAnalytics()
    ]);
    
    return { users, posts, analytics };
  } catch (error) {
    console.error('Error loading dashboard:', error);
  }
}
```

## 🛠️ Frameworks y Librerías Populares

### React Moderno

#### Hooks Esenciales

```jsx
import React, { useState, useEffect, useCallback, useMemo } from 'react';

function UserDashboard({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // useEffect para carga de datos
  useEffect(() => {
    async function loadUser() {
      try {
        setLoading(true);
        const userData = await fetchUserData(userId);
        setUser(userData);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    loadUser();
  }, [userId]);

  // useCallback para funciones que se pasan como props
  const handleUserUpdate = useCallback(async (updates) => {
    try {
      const updatedUser = await updateUser(userId, updates);
      setUser(updatedUser);
    } catch (err) {
      setError(err.message);
    }
  }, [userId]);

  // useMemo para cálculos pesados
  const userStats = useMemo(() => {
    if (!user) return null;
    
    return {
      totalPosts: user.posts?.length || 0,
      averageRating: user.posts?.reduce((acc, post) => acc + post.rating, 0) / user.posts?.length || 0
    };
  }, [user]);

  if (loading) return <div>Cargando...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="user-dashboard">
      <h1>{user.name}</h1>
      <UserProfile user={user} onUpdate={handleUserUpdate} />
      <UserStats stats={userStats} />
    </div>
  );
}
```

#### Custom Hooks

```jsx
// Hook personalizado para manejo de formularios
function useForm(initialValues, validationSchema) {
  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = useCallback((name, value) => {
    setValues(prev => ({ ...prev, [name]: value }));
    
    // Limpiar error cuando el usuario empieza a escribir
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: null }));
    }
  }, [errors]);

  const validate = useCallback(() => {
    const newErrors = {};
    
    Object.keys(validationSchema).forEach(field => {
      const rules = validationSchema[field];
      const value = values[field];
      
      rules.forEach(rule => {
        if (!rule.test(value)) {
          newErrors[field] = rule.message;
        }
      });
    });
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [values, validationSchema]);

  const handleSubmit = useCallback(async (onSubmit) => {
    if (!validate()) return;
    
    setIsSubmitting(true);
    try {
      await onSubmit(values);
    } catch (error) {
      console.error('Form submission error:', error);
    } finally {
      setIsSubmitting(false);
    }
  }, [values, validate]);

  return {
    values,
    errors,
    isSubmitting,
    handleChange,
    handleSubmit
  };
}
```

### Vue.js 3 Composition API

```vue
<template>
  <div class="todo-app">
    <h1>Lista de Tareas</h1>
    
    <form @submit.prevent="addTodo">
      <input
        v-model="newTodoText"
        placeholder="Nueva tarea..."
        required
      />
      <button type="submit" :disabled="!newTodoText.trim()">
        Agregar
      </button>
    </form>

    <ul>
      <li
        v-for="todo in filteredTodos"
        :key="todo.id"
        :class="{ completed: todo.completed }"
      >
        <input
          type="checkbox"
          v-model="todo.completed"
        />
        <span>{{ todo.text }}</span>
        <button @click="removeTodo(todo.id)">Eliminar</button>
      </li>
    </ul>

    <div class="filters">
      <button
        v-for="filter in filters"
        :key="filter"
        :class="{ active: currentFilter === filter }"
        @click="currentFilter = filter"
      >
        {{ filter }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue';

// Estado reactivo
const todos = ref([]);
const newTodoText = ref('');
const currentFilter = ref('Todas');
const filters = ['Todas', 'Pendientes', 'Completadas'];

// Computed properties
const filteredTodos = computed(() => {
  switch (currentFilter.value) {
    case 'Pendientes':
      return todos.value.filter(todo => !todo.completed);
    case 'Completadas':
      return todos.value.filter(todo => todo.completed);
    default:
      return todos.value;
  }
});

// Métodos
function addTodo() {
  const text = newTodoText.value.trim();
  if (!text) return;

  todos.value.push({
    id: Date.now(),
    text,
    completed: false,
    createdAt: new Date()
  });

  newTodoText.value = '';
}

function removeTodo(id) {
  const index = todos.value.findIndex(todo => todo.id === id);
  if (index > -1) {
    todos.value.splice(index, 1);
  }
}

// Lifecycle hooks
onMounted(() => {
  // Cargar todos del localStorage
  const savedTodos = localStorage.getItem('todos');
  if (savedTodos) {
    todos.value = JSON.parse(savedTodos);
  }
});

// Watcher para guardar en localStorage
watch(todos, (newTodos) => {
  localStorage.setItem('todos', JSON.stringify(newTodos));
}, { deep: true });
</script>
```

## 🏗️ Herramientas de Desarrollo

### Vite - Build Tool Moderno

```javascript
// vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

export default defineConfig({
  plugins: [react()],
  
  // Alias para imports más limpios
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@utils': resolve(__dirname, 'src/utils'),
      '@styles': resolve(__dirname, 'src/styles')
    }
  },
  
  // Variables de entorno
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version)
  },
  
  // Configuración del servidor de desarrollo
  server: {
    port: 3000,
    open: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  },
  
  // Optimización de build
  build: {
    target: 'esnext',
    minify: 'esbuild',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          router: ['react-router-dom'],
          ui: ['@mui/material', '@emotion/react']
        }
      }
    }
  }
});
```

### ESLint y Prettier Configuración

```javascript
// .eslintrc.js
module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: true
  },
  extends: [
    'eslint:recommended',
    '@typescript-eslint/recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'prettier'
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaFeatures: {
      jsx: true
    },
    ecmaVersion: 'latest',
    sourceType: 'module'
  },
  plugins: [
    'react',
    '@typescript-eslint',
    'react-hooks'
  ],
  rules: {
    'react/react-in-jsx-scope': 'off',
    'react/prop-types': 'off',
    '@typescript-eslint/no-unused-vars': 'error',
    'prefer-const': 'error',
    'no-var': 'error'
  },
  settings: {
    react: {
      version: 'detect'
    }
  }
};
```

```json
// .prettierrc
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 80,
  "tabWidth": 2,
  "useTabs": false,
  "bracketSpacing": true,
  "arrowParens": "avoid"
}
```

## 🎯 Testing Moderno

### Vitest para Unit Testing

```javascript
// sum.test.js
import { describe, test, expect } from 'vitest';
import { sum, multiply, divide } from './math-utils';

describe('Math Utils', () => {
  test('suma dos números correctamente', () => {
    expect(sum(2, 3)).toBe(5);
    expect(sum(-1, 1)).toBe(0);
    expect(sum(0, 0)).toBe(0);
  });

  test('multiplica dos números correctamente', () => {
    expect(multiply(3, 4)).toBe(12);
    expect(multiply(-2, 3)).toBe(-6);
    expect(multiply(0, 5)).toBe(0);
  });

  test('divide dos números correctamente', () => {
    expect(divide(10, 2)).toBe(5);
    expect(divide(7, 2)).toBe(3.5);
    
    // Test de error
    expect(() => divide(5, 0)).toThrow('División por cero no permitida');
  });
});
```

### React Testing Library

```jsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, test, expect, vi } from 'vitest';
import UserProfile from './UserProfile';

describe('UserProfile Component', () => {
  const mockUser = {
    id: 1,
    name: 'Juan Pérez',
    email: 'juan@example.com',
    role: 'admin'
  };

  test('renderiza información del usuario', () => {
    render(<UserProfile user={mockUser} />);
    
    expect(screen.getByText('Juan Pérez')).toBeInTheDocument();
    expect(screen.getByText('juan@example.com')).toBeInTheDocument();
    expect(screen.getByText('admin')).toBeInTheDocument();
  });

  test('llama onEdit cuando se hace clic en editar', async () => {
    const mockOnEdit = vi.fn();
    
    render(<UserProfile user={mockUser} onEdit={mockOnEdit} />);
    
    const editButton = screen.getByRole('button', { name: /editar/i });
    fireEvent.click(editButton);
    
    await waitFor(() => {
      expect(mockOnEdit).toHaveBeenCalledWith(mockUser);
    });
  });

  test('muestra loading state', () => {
    render(<UserProfile user={null} loading={true} />);
    
    expect(screen.getByText(/cargando/i)).toBeInTheDocument();
  });
});
```

## 🔗 Recursos Adicionales

### Herramientas Recomendadas

- **Bundlers**: Vite, Webpack 5, Rollup
- **Testing**: Vitest, Jest, Cypress
- **State Management**: Zustand, Redux Toolkit, Jotai
- **Styling**: Tailwind CSS, Styled Components, Emotion
- **Type Safety**: TypeScript, PropTypes

### Librerías Útiles

```javascript
// Gestión de fechas
import { format, parseISO, isAfter } from 'date-fns';

// HTTP requests
import axios from 'axios';

// Validación de formularios
import { z } from 'zod';

// Utilidades
import { debounce, throttle } from 'lodash-es';

// Animations
import { motion } from 'framer-motion';
```

### Mejores Prácticas

1. **Usa TypeScript** para mayor seguridad de tipos
2. **Implementa testing** desde el principio
3. **Optimiza el rendimiento** con lazy loading y code splitting
4. **Mantén componentes pequeños** y enfocados en una responsabilidad
5. **Usa linting y formatting** para consistencia de código
6. **Implementa CI/CD** para despliegues automáticos
