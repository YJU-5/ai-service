FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

ENV S3_BUCKET_NAME=your-model-bucket
ENV S3_MODEL_KEY=profanity_filter_model.zip
ENV AWS_DEFAULT_REGION=ap-northeast-2

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 