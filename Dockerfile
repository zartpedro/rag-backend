FROM python:3.13-slim

RUN pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .

# ---- LINHA DE TESTE MODIFICADA ----
RUN echo "INFO: Tentando instalar azure-ai-openai diretamente do PyPI público com log detalhado..." && \
  python -m pip install --index-url https://pypi.org/simple -vvv azure-ai-openai && \
  echo "INFO: Instalação direta de azure-ai-openai BEM-SUCEDIDA." || \
  (echo "ERRO: Falha na instalação direta de azure-ai-openai." && exit 1)
# ---- FIM DA LINHA DE TESTE ----

RUN echo "INFO: Instalando pacotes a partir do requirements.txt..." && \
  pip install --no-cache-dir --pre -r requirements.txt \
  && pip install --upgrade pip setuptools wheel && \
  echo "INFO: Pacotes do requirements.txt instalados."

COPY . .
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:80", "--workers", "4"]