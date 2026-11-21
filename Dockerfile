FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py database.py features.py analytics.py ./
COPY models ./models
COPY data/raw ./data/raw

EXPOSE 8002

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8002}"]
