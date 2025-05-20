FROM python:3.10-slim

# Instala dependências de SO (se precisar de libs para PDF etc)
RUN apt-get update && \
  apt-get install -y gcc build-essential libxml2-dev libxslt-dev && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expõe a porta padrão do container (80)
EXPOSE 80

# Inicia com uvicorn via gunicorn para produção
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:80", "--workers", "2"]
