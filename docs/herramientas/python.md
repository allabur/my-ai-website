---
sidebar_position: 1
---

# Python para IA y Data Science

Python se ha convertido en el lenguaje dominante para inteligencia artificial y ciencia de datos. En esta sección, exploraremos las herramientas y librerías más importantes.

## 🐍 ¿Por qué Python para IA?

### Ventajas de Python

- **Sintaxis simple y legible**: Ideal para prototipado rápido
- **Ecosistema rico**: Miles de librerías especializadas
- **Comunidad activa**: Gran soporte y documentación
- **Integración**: Fácil de integrar con otros lenguajes y sistemas
- **Performance**: Librerías optimizadas en C/C++

## 📚 Librerías Esenciales

### Para Machine Learning

#### Scikit-learn
La librería más popular para ML tradicional:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ejemplo rápido
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

#### TensorFlow/Keras
Para deep learning:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Red neuronal simple
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Para Análisis de Datos

#### Pandas
Manipulación y análisis de datos:

```python
import pandas as pd

# Cargar datos
df = pd.read_csv('datos.csv')

# Análisis exploratorio
print(df.describe())
print(df.info())

# Limpieza de datos
df_clean = df.dropna().drop_duplicates()

# Agrupaciones
resultado = df.groupby('categoria').agg({
    'ventas': 'sum',
    'precio': 'mean'
})
```

#### NumPy
Computación numérica:

```python
import numpy as np

# Operaciones vectorizadas
arr = np.array([1, 2, 3, 4, 5])
resultado = np.sqrt(arr) * 2

# Álgebra lineal
matriz_a = np.random.rand(3, 3)
matriz_b = np.random.rand(3, 3)
producto = np.dot(matriz_a, matriz_b)
```

### Para Visualización

#### Matplotlib
Gráficos básicos:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='Serie 1')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.title('Mi Gráfico')
plt.legend()
plt.show()
```

#### Seaborn
Visualizaciones estadísticas:

```python
import seaborn as sns

# Heatmap de correlación
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Distribuciones
sns.distplot(df['column'])
plt.show()
```

## 🛠️ Herramientas de Desarrollo

### Jupyter Notebooks

```bash
# Instalación
pip install jupyter

# Ejecutar
jupyter notebook
```

**Ventajas**:
- Desarrollo interactivo
- Visualizaciones integradas
- Documentación en línea
- Fácil experimentación

### IDEs Recomendados

1. **PyCharm**: IDE completo con debugging avanzado
2. **VS Code**: Ligero con excelentes extensiones
3. **JupyterLab**: Entorno web avanzado

## 🔬 Flujo de Trabajo Típico

### 1. Exploración de Datos

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv('dataset.csv')

# Exploración inicial
print(f"Shape: {df.shape}")
print(f"Columnas: {df.columns.tolist()}")
print(f"Tipos de datos:\n{df.dtypes}")
print(f"Valores nulos:\n{df.isnull().sum()}")

# Estadísticas descriptivas
df.describe()
```

### 2. Preprocesamiento

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Limpiar datos
df_clean = df.dropna()

# Codificar variables categóricas
le = LabelEncoder()
df_clean['categoria_encoded'] = le.fit_transform(df_clean['categoria'])

# Normalizar features numéricas
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_clean[['feature1', 'feature2']])
```

### 3. Modelado

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Dividir datos
X = df_clean[features]
y = df_clean['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### 4. Evaluación

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Predicciones
y_pred = model.predict(X_test)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.show()

# Reporte detallado
print(classification_report(y_test, y_pred))
```

## 🚀 Mejores Prácticas

### Estructura de Proyecto

```
proyecto-ia/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── exploratory/
│   └── reports/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── models/
├── reports/
├── requirements.txt
└── README.md
```

### Manejo de Dependencias

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Generar requirements
pip freeze > requirements.txt
```

### Versionado de Modelos

```python
import joblib
from datetime import datetime

# Guardar modelo
model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
joblib.dump(model, f"models/{model_name}")

# Cargar modelo
model_loaded = joblib.load("models/model_20240101_120000.pkl")
```

## 📊 Ejemplo Completo: Predicción de Precios

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Cargar y explorar datos
df = pd.read_csv('house_prices.csv')
print(df.head())

# 2. Preprocesamiento
# Seleccionar features numéricas
numeric_features = df.select_dtypes(include=[np.number]).columns
X = df[numeric_features].drop('price', axis=1)
y = df['price']

# Limpiar datos
X = X.fillna(X.mean())

# 3. División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Entrenamiento
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predicción y evaluación
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.3f}")

# 6. Visualización
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Precio Real')
plt.ylabel('Precio Predicho')
plt.title('Predicción vs Realidad')
plt.show()

# 7. Importancia de features
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
```

## 🔗 Recursos Adicionales

### Cursos Recomendados
- [Curso de Python para Data Science](https://www.coursera.org/learn/python-for-applied-data-science-ai)
- [Machine Learning con Python](https://www.edx.org/course/machine-learning-with-python)

### Libros
- "Python Machine Learning" - Sebastian Raschka
- "Hands-On Machine Learning" - Aurélien Géron
- "Python for Data Analysis" - Wes McKinney

### Comunidades
- [Stack Overflow](https://stackoverflow.com/questions/tagged/python)
- [Reddit r/MachineLearning](https://reddit.com/r/MachineLearning)
- [Kaggle](https://kaggle.com)
