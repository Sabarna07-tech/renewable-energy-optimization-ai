FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY serving/requirements.txt /app/serving/requirements.txt
RUN pip install -r /app/serving/requirements.txt
COPY serving /app/serving
COPY ml/models/xgboost_model.pkl /app/ml/models/xgboost_model.pkl
EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
