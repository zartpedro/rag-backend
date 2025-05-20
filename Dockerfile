FROM python:3.10

# 1) Atualiza pip para a versão mais recente
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .

# 2) Instala as dependências, permitindo pre-releases
RUN pip install --no-cache-dir --pre -r requirements.txt

COPY . .
EXPOSE 80
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:80", "--workers", "2"]
