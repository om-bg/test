# --- Étape 1: Builder pour compiler les dépendances ---
# Utilise une image Python slim-buster pour minimiser la taille de base.
FROM python:3.9-slim-buster as builder

# Définit l'encodage et s'assure que la sortie Python n'est pas mise en tampon.
ENV LANG C.UTF-8
ENV PYTHONUNBUFFERED 1

# Installe les dépendances système nécessaires pour la compilation d'OpenCV
# et d'autres bibliothèques C++ utilisées par NumPy ou ONNX Runtime.
# `--no-install-recommends` réduit les paquets inutiles.
# Nettoie le cache apt pour réduire la taille.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        pkg-config \
        libatlas-base-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        zlib1g-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libgl1-mesa-glx \
        libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Définit le répertoire de travail à l'intérieur du conteneur.
WORKDIR /app

# Copie le fichier requirements.txt en premier pour tirer parti de la mise en cache de Docker.
COPY requirements.txt .

# Installe les dépendances Python.
# `--no-cache-dir` empêche pip de stocker les packages téléchargés, réduisant la taille.
# `opencv-python-headless` est crucial pour éviter l'interface graphique.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Étape 2: Image Finale pour l'exécution ---
# Utilise à nouveau une image slim-buster pour la légèreté.
FROM python:3.9-slim-buster

# Définit l'encodage et les variables d'environnement pour l'exécution.
ENV LANG C.UTF-8
ENV PYTHONUNBUFFERED 1

# Définit le répertoire de travail.
WORKDIR /app

# Copie uniquement les packages Python installés et les binaires de la première étape
# vers cette étape finale. Cela exclut tous les outils de compilation de la première étape.
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copie tous les autres fichiers de votre application (y compris main.py et yolov8n.onnx)
# de votre machine locale vers le répertoire de travail du conteneur.
COPY . .

# Expose le port sur lequel FastAPI va écouter.
EXPOSE 8000

# Commande de démarrage de l'application avec Uvicorn.
# Assurez-vous que 'main:app' correspond au nom de votre fichier et de votre instance FastAPI.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
