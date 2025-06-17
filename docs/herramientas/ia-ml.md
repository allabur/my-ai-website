---
sidebar_position: 3
---

# Herramientas de IA y Machine Learning

El ecosistema de herramientas para inteligencia artificial y machine learning está en constante evolución. Aquí te presento las herramientas más importantes y efectivas que uso en mis proyectos.

## 🧠 Frameworks de Deep Learning

### TensorFlow 2.x

TensorFlow sigue siendo uno de los frameworks más populares y robustos para deep learning.

#### Instalación y Configuración

```bash
# CPU version
pip install tensorflow

# GPU version (requires CUDA)
pip install tensorflow-gpu

# Verificar instalación
python -c "import tensorflow as tf; print(tf.__version__)"
```

#### Ejemplo: Red Neuronal Simple

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Preparar datos
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Crear modelo
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compilar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluar
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
```

#### Transfer Learning con TensorFlow

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Cargar modelo preentrenado
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Congelar capas base
base_model.trainable = False

# Añadir capas personalizadas
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compilar con learning rate bajo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning
base_model.trainable = True
fine_tune_at = 100

# Congelar hasta la capa fine_tune_at
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001/10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### PyTorch

PyTorch es excelente para investigación y desarrollo rápido de prototipos.

#### Ejemplo: CNN para Clasificación de Imágenes

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Primera capa convolucional
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Segunda capa convolucional
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Tercera capa convolucional
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Configuración de entrenamiento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Loop de entrenamiento
def train_model(model, train_loader, val_loader, num_epochs=25):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Fase de entrenamiento
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Epoch {epoch}/{num_epochs-1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        scheduler.step()
    
    return model
```

## 🤗 Hugging Face Transformers

La librería Transformers es esencial para NLP moderno.

### Análisis de Sentimientos

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Pipeline simple
sentiment_pipeline = pipeline("sentiment-analysis")
results = sentiment_pipeline([
    "Me encanta este producto!",
    "No me gustó nada la experiencia"
])

print(results)

# Modelo específico en español
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)
```

### Generación de Texto

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextGenerationPipeline

# Cargar modelo GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Configurar padding token
tokenizer.pad_token = tokenizer.eos_token

# Pipeline de generación
generator = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Generar texto
prompt = "La inteligencia artificial es"
generated = generator(
    prompt,
    max_length=100,
    num_return_sequences=3,
    temperature=0.8,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

for i, text in enumerate(generated):
    print(f"Generación {i+1}: {text['generated_text']}")
```

### Fine-tuning de Modelos

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset
import torch

# Preparar datos
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=512
    )

# Cargar modelo y tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Tokenizar datos
train_dataset = Dataset.from_dict({
    'text': train_texts,
    'labels': train_labels
})
train_dataset = train_dataset.map(tokenize_function, batched=True)

# Configurar entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Entrenar
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## 📊 MLOps y Gestión de Experimentos

### MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configurar MLflow
mlflow.set_experiment("mi-experimento-ml")

with mlflow.start_run():
    # Hiperparámetros
    n_estimators = 100
    max_depth = 10
    random_state = 42
    
    # Log de parámetros
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    
    # Entrenar modelo
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Predicciones
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Log de métricas
    mlflow.log_metric("accuracy", accuracy)
    
    # Log del modelo
    mlflow.sklearn.log_model(model, "modelo")
    
    # Log de artefactos
    with open("classification_report.txt", "w") as f:
        f.write(classification_report(y_test, predictions))
    mlflow.log_artifact("classification_report.txt")
```

### Weights & Biases (wandb)

```python
import wandb
import torch
import torch.nn as nn

# Inicializar wandb
wandb.init(
    project="mi-proyecto-dl",
    config={
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
        "architecture": "CNN"
    }
)

config = wandb.config

# Definir modelo
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Loop de entrenamiento con logging
for epoch in range(config.epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log cada 100 batches
        if batch_idx % 100 == 0:
            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })
    
    # Log de métricas por época
    avg_loss = total_loss / len(train_loader)
    val_accuracy = evaluate_model(model, val_loader)
    
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_loss,
        "val_accuracy": val_accuracy
    })

# Guardar modelo
wandb.save("modelo.pth")
```

## 🔍 Herramientas de Análisis de Datos

### Pandas Avanzado

```python
import pandas as pd
import numpy as np

# Lectura eficiente de datos grandes
df = pd.read_csv(
    'large_dataset.csv',
    chunksize=10000,  # Leer en chunks
    dtype={'column1': 'category'},  # Optimizar tipos
    parse_dates=['date_column']
)

# Procesamiento por chunks
processed_chunks = []
for chunk in df:
    # Procesamiento del chunk
    processed_chunk = chunk.groupby('category').agg({
        'value': ['mean', 'std', 'count'],
        'date_column': ['min', 'max']
    })
    processed_chunks.append(processed_chunk)

# Combinar resultados
final_result = pd.concat(processed_chunks).groupby(level=0).sum()

# Pipeline de transformación
def data_pipeline(df):
    return (df
            .pipe(clean_data)
            .pipe(feature_engineering)
            .pipe(normalize_features)
            .pipe(select_features))

def clean_data(df):
    # Remover outliers usando IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    filter_mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[filter_mask]

def feature_engineering(df):
    # Crear nuevas features
    df['feature_ratio'] = df['feature1'] / df['feature2']
    df['feature_interaction'] = df['feature1'] * df['feature3']
    
    # Encoding categórico
    df = pd.get_dummies(df, columns=['categorical_column'])
    
    return df
```

### Plotly para Visualizaciones Interactivas

```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Dashboard interactivo
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Distribución', 'Serie Temporal', 'Correlación', 'Scatter Plot'),
    specs=[[{"secondary_y": True}, {"type": "xy"}],
           [{"type": "xy"}, {"type": "scatter"}]]
)

# Histograma
fig.add_trace(
    go.Histogram(x=df['column1'], name='Distribución'),
    row=1, col=1
)

# Serie temporal
fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['value'],
        mode='lines',
        name='Tendencia'
    ),
    row=1, col=2
)

# Heatmap de correlación
correlation_matrix = df.corr()
fig.add_trace(
    go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu'
    ),
    row=2, col=1
)

# Scatter plot
fig.add_trace(
    go.Scatter(
        x=df['feature1'],
        y=df['feature2'],
        mode='markers',
        marker=dict(
            size=df['feature3'],
            color=df['target'],
            colorscale='Viridis',
            showscale=True
        ),
        name='Scatter'
    ),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=False, title_text="Dashboard ML")
fig.show()
```

## 🚀 Herramientas de Productividad

### AutoML con PyCaret

```python
import pycaret
from pycaret.classification import *

# Configurar entorno
clf = setup(
    data=df,
    target='target_column',
    session_id=123,
    train_size=0.8,
    fold_strategy='stratifiedkfold',
    fold=5
)

# Comparar modelos
best_models = compare_models(
    include=['rf', 'xgboost', 'lightgbm', 'catboost'],
    sort='Accuracy',
    n_select=3
)

# Crear modelo específico
rf_model = create_model('rf')

# Tuning de hiperparámetros
tuned_rf = tune_model(rf_model, optimize='Accuracy')

# Evaluar modelo
evaluate_model(tuned_rf)

# Finalizar y desplegar
final_model = finalize_model(tuned_rf)
deploy_model(final_model, model_name='mi_modelo_rf', platform='aws')
```

### Optuna para Optimización de Hiperparámetros

```python
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score

def objective(trial):
    # Definir espacio de búsqueda
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    
    # Crear y evaluar modelo
    model = xgb.XGBClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    return scores.mean()

# Optimizar
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Mejores parámetros
best_params = study.best_params
print(f"Best parameters: {best_params}")
print(f"Best value: {study.best_value}")

# Visualizar optimización
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

## 🔗 Recursos y Herramientas Adicionales

### Datasets y APIs

```python
# Datasets populares
import datasets

# Cargar dataset de Hugging Face
dataset = datasets.load_dataset("imdb")

# Kaggle datasets
import kaggle
kaggle.api.dataset_download_files('dataset-name', path='./data')

# APIs útiles
import requests

# OpenAI API
import openai
openai.api_key = "tu-api-key"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Explica machine learning"}]
)
```

### Herramientas de Monitoreo

```python
# Evidently AI para drift detection
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

column_mapping = ColumnMapping()
data_drift_report = Report(metrics=[DataDriftPreset()])

data_drift_report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

data_drift_report.show(mode='inline')
```

## 📚 Recursos de Aprendizaje

### Cursos Recomendados
- [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [CS231n Stanford](http://cs231n.stanford.edu/)

### Libros Esenciales
- "Hands-On Machine Learning" - Aurélien Géron
- "Deep Learning" - Ian Goodfellow
- "Pattern Recognition and Machine Learning" - Christopher Bishop

### Comunidades
- [Kaggle](https://kaggle.com)
- [Papers with Code](https://paperswithcode.com)
- [Reddit r/MachineLearning](https://reddit.com/r/MachineLearning)
