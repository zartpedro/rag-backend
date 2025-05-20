FROM python:3.13-slim

RUN pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir --pre -r requirements.txt \
  && pip install --upgrade pip setuptools wheel

COPY . .

CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:80", "--workers", "4"]
