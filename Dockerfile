FROM python:3.13-slim

RUN pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .

# ---- LINHA DE TESTE IMPORTANTE ----
RUN echo "Tentando instalar azure-ai-openai diretamente..." && \
  python -m pip install azure-ai-openai && \
  echo "Instalação direta de azure-ai-openai bem-sucedida." || \
  (echo "Falha na instalação direta de azure-ai-openai." && exit 1)
# ---- FIM DA LINHA DE TESTE ----

RUN pip install --no-cache-dir --pre -r requirements.txt \
  && pip install --upgrade pip setuptools wheel

COPY . .

CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:80", "--workers", "4"]
