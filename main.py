import logging
import os
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import boto3
import torch
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

load_dotenv()

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv('MODEL_DIR', './profanity_filter_model'))
MODEL_ARCHIVE = Path(os.getenv('MODEL_ARCHIVE', './model.zip'))
PROFANITY_THRESHOLD = float(os.getenv('PROFANITY_THRESHOLD', '0.5'))

model: DistilBertForSequenceClassification | None = None
tokenizer: DistilBertTokenizerFast | None = None


def parse_cors_origins() -> list[str]:
    raw_origins = os.getenv(
        'CORS_ORIGINS',
        'http://localhost:3001,http://localhost:3005',
    )
    return [
        origin.strip()
        for origin in raw_origins.split(',')
        if origin.strip()
    ]


def extract_model_archive(archive: Path, destination: Path) -> None:
    destination_root = destination.resolve()

    with zipfile.ZipFile(archive, 'r') as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError('The model archive contains an unsafe path')

        zip_file.extractall(destination)


def download_model_from_s3() -> None:
    bucket_name = os.getenv('S3_BUCKET_NAME')
    model_key = os.getenv('S3_MODEL_KEY', 'profanity_filter_model.zip')

    if not bucket_name:
        raise RuntimeError('S3_BUCKET_NAME is required when no local model exists')

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Downloading model artifact from S3')

    try:
        boto3.client('s3').download_file(
            bucket_name,
            model_key,
            str(MODEL_ARCHIVE),
        )
        extract_model_archive(MODEL_ARCHIVE, MODEL_DIR.parent)
    except (BotoCoreError, ClientError, OSError, zipfile.BadZipFile) as error:
        raise RuntimeError('Failed to download or extract the model') from error
    finally:
        MODEL_ARCHIVE.unlink(missing_ok=True)


def initialize_model() -> None:
    global model, tokenizer

    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        download_model_from_s3()

    logger.info('Loading tokenizer and classification model')
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    logger.info('Model initialization completed')


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_model()
    yield


app = FastAPI(
    title='Let Eat Go AI Service',
    description='Text classification service for community safety',
    version='1.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(),
    allow_credentials=True,
    allow_methods=['GET', 'POST'],
    allow_headers=['Content-Type', 'Authorization'],
)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=512)


class PredictionResponse(BaseModel):
    is_profanity: bool
    confidence: float


@app.post('/predict', response_model=PredictionResponse)
async def predict(request: TextRequest) -> PredictionResponse:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail='Model is not initialized')

    try:
        inputs = tokenizer(
            request.text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.inference_mode():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            profanity_probability = probabilities[0][1].item()

        return PredictionResponse(
            is_profanity=profanity_probability >= PROFANITY_THRESHOLD,
            confidence=profanity_probability,
        )
    except Exception as error:
        logger.exception('Prediction failed')
        raise HTTPException(status_code=500, detail='Prediction failed') from error


@app.get('/health')
async def health_check() -> dict[str, bool | str]:
    model_loaded = model is not None and tokenizer is not None
    return {
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
    }
