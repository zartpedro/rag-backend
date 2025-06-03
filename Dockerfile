FROM python:3.10-slim

WORKDIR /app

# Copia somente o requirements.txt primeiro (para cache do Docker)
COPY requirements.txt /app/requirements.txt

# Instala as dependências listadas
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código fonte
COPY . /app

# Expõe a porta em que o Gunicorn/Uvicorn irá escutar
EXPOSE 80

# O comando abaixo assume que você tenha algo como "gunicorn.conf.py" configurado
# e que o seu app FastAPI se chame "main:app"
CMD ["gunicorn", "main:app", "-c", "gunicorn.conf.py"]
