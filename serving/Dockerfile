FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY src/ ./src/
COPY serving/ ./serving/

ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_REGISTERED_MODEL_NAME=image-captioner

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
