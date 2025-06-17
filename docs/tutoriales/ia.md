# Inteligencia Artificial

Guías y tutoriales sobre IA, Machine Learning y Deep Learning.

## 🧠 Introducción a la IA

### ¿Qué es la Inteligencia Artificial?

La IA es una rama de la informática que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.

### Tipos de IA

1. **IA Estrecha (ANI)**
   - Sistemas específicos para tareas concretas
   - Reconocimiento de voz, visión por computadora
   - La mayoría de sistemas actuales

2. **IA General (AGI)**
   - Sistemas con capacidades cognitivas humanas
   - Aún en desarrollo teórico
   - Objetivo a largo plazo

## 🤖 Machine Learning

### Tipos de Aprendizaje

- **Supervisado**: Datos etiquetados
- **No supervisado**: Patrones en datos sin etiquetar
- **Reforzamiento**: Aprendizaje por recompensas

### Algoritmos Populares

1. **Clasificación**
   - Random Forest
   - Support Vector Machines
   - Redes Neuronales

2. **Regresión**
   - Regresión Lineal
   - Regresión Polinomial
   - Ridge/Lasso

## 🧠 Deep Learning

### Redes Neuronales

- **Perceptrón**: La unidad básica
- **Redes densas**: Capas completamente conectadas
- **CNNs**: Para procesamiento de imágenes
- **RNNs/LSTMs**: Para secuencias temporales

### Frameworks Populares

- **TensorFlow**: De Google
- **PyTorch**: De Meta
- **Keras**: API de alto nivel
- **Scikit-learn**: Para ML tradicional

## 🛠️ Herramientas y Recursos

### Entornos de Desarrollo
- Jupyter Notebooks
- Google Colab
- Kaggle Kernels

### Datasets
- Kaggle
- UCI ML Repository
- OpenML
- Papers with Code

## 🎯 Proyecto Práctico

### Clasificador de Imágenes

```python
import tensorflow as tf
from tensorflow import keras

# Cargar datos
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalizar
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Crear modelo
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compilar
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

## 📚 Recursos de Aprendizaje

- **Cursos**: Coursera ML, Fast.ai, CS229
- **Libros**: "Hands-On ML" por Aurélien Géron
- **Papers**: arXiv, Papers with Code
- **Comunidades**: Reddit r/MachineLearning, Stack Overflow

---

*Última actualización: Junio 2025*
