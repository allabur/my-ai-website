---
slug: como-crear-api-rest-python-fastapi
title: Cómo Crear una API REST Moderna con Python y FastAPI
authors: [tu-nombre]
tags: [python, fastapi, api, backend, tutorial]
date: 2024-02-01
---

# Cómo Crear una API REST Moderna con Python y FastAPI

FastAPI se ha convertido en uno de los frameworks más populares para crear APIs en Python. Su sintaxis intuitiva, documentación automática y alto rendimiento lo hacen ideal tanto para principiantes como para proyectos empresariales.

<!-- truncate -->

## ¿Por qué FastAPI?

FastAPI combina lo mejor de varios mundos:

- **Velocidad**: Uno de los frameworks más rápidos disponibles
- **Fácil de usar**: Sintaxis intuitiva basada en type hints de Python
- **Documentación automática**: Genera documentación interactiva automáticamente
- **Validación automática**: Validación de datos basada en tipos Python
- **Estándares modernos**: Compatible con OpenAPI y JSON Schema

## Configuración del Proyecto

### Instalación

Primero, creemos un entorno virtual e instalemos las dependencias:

```bash
# Crear entorno virtual
python -m venv fastapi-env

# Activar entorno virtual (Windows)
fastapi-env\Scripts\activate

# Instalar dependencias
pip install fastapi uvicorn[standard] sqlalchemy psycopg2-binary
```

### Estructura del Proyecto

```
mi-api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   └── database.py
├── requirements.txt
└── README.md
```

## Creando Nuestra Primera API

### Archivo Principal (main.py)

```python
from fastapi import FastAPI, HTTPException, Depends
from typing import List
import uvicorn

app = FastAPI(
    title="Mi API REST",
    description="Una API moderna con FastAPI",
    version="1.0.0"
)

# Base de datos simulada
users_db = []

class User(BaseModel):
    id: int
    name: str
    email: str
    is_active: bool = True

@app.get("/")
async def root():
    return {"message": "¡Bienvenido a mi API!"}

@app.get("/users", response_model=List[User])
async def get_users():
    return users_db

@app.post("/users", response_model=User)
async def create_user(user: User):
    users_db.append(user)
    return user

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    for user in users_db:
        if user.id == user_id:
            return user
    raise HTTPException(status_code=404, detail="Usuario no encontrado")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Modelos de Datos (schemas.py)

```python
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    name: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        orm_mode = True
```

## Integrando Base de Datos

### Configuración de SQLAlchemy (database.py)

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/dbname"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Modelos de Base de Datos (models.py)

```python
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

## Características Avanzadas

### Autenticación JWT

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY = "tu-clave-secreta-super-segura"
ALGORITHM = "HS256"

security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Token inválido")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido")

@app.get("/protected")
async def protected_route(current_user: str = Depends(verify_token)):
    return {"message": f"Hola {current_user}, esta es una ruta protegida"}
```

### Middleware de CORS

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Permitir React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Validación Avanzada

```python
from pydantic import validator
from typing import Optional

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    age: Optional[int] = None

    @validator('name')
    def name_must_contain_space(cls, v):
        if ' ' not in v:
            raise ValueError('El nombre debe contener nombre y apellido')
        return v.title()

    @validator('age')
    def age_must_be_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError('La edad debe ser positiva')
        return v
```

## Testing

```python
import pytest
from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "¡Bienvenido a mi API!"}

def test_create_user():
    response = client.post(
        "/users",
        json={"name": "Juan Pérez", "email": "juan@example.com"}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Juan Pérez"
```

## Despliegue

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Requirements.txt

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
```

## Mejores Prácticas

### 1. Estructura del Proyecto
- Separa lógica de negocio de la lógica de API
- Usa dependency injection
- Implementa logging apropiado

### 2. Seguridad
- Nunca hardcodees secrets
- Implementa rate limiting
- Usa HTTPS en producción

### 3. Performance
- Implementa caching cuando sea apropiado
- Usa connection pooling para la base de datos
- Monitora performance con herramientas como Prometheus

## Conclusión

FastAPI es una excelente opción para crear APIs modernas en Python. Su combinación de velocidad, facilidad de uso y características avanzadas lo convierte en una herramienta poderosa para cualquier desarrollador.

El código completo de este tutorial está disponible en mi [GitHub](https://github.com/tu-usuario/fastapi-tutorial).

¿Tienes preguntas sobre FastAPI? ¡Déjalas en los comentarios!

---

**Próximo artículo**: "Implementando GraphQL con FastAPI y Strawberry" 🍓
