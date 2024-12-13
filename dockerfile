FROM python:3.10-slim

# Crear directorio de la app
WORKDIR /app

# Instalar dependencias del sistema si es necesario
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de la app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponer el puerto 5000
EXPOSE 5000

CMD ["python", "app.py"]
