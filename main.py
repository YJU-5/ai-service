# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import os

# app = FastAPI()

# # CORS 설정 - 다른 도메인에서의 API 접근을 허용
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 실제 운영 환경에서는 특정 도메인만 허용하도록 수정 필요
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 한국어 BERT 모델 초기화
# # klue/bert-base는 한국어에 특화된 BERT 모델 (일본어 특화로 수정)
# MODEL_PATH = "klue/bert-base"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)  # 2개의 레이블: 욕설(1) / 비욕설(0)

# # API 요청 데이터 모델 정의
# class TextRequest(BaseModel):
#     text: str  # 검사할 텍스트

# @app.post("/predict")
# async def predict(request: TextRequest):
#     """
#     텍스트의 욕설 여부를 검사하는 엔드포인트
    
#     Args:
#         request (TextRequest): 검사할 텍스트를 포함한 요청 객체
    
#     Returns:
#         dict: {
#             "is_profanity": bool,  # 욕설 포함 여부
#             "confidence": float    # 예측 신뢰도 (0.0 ~ 1.0)
#         }
#     """
#     try:
#         # 텍스트를 모델이 이해할 수 있는 형태로 변환
#         inputs = tokenizer(
#             request.text,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=512  # 최대 길이 제한
#         )
        
#         # 모델을 통한 예측 수행
#         with torch.no_grad():  # 추론 시에는 그래디언트 계산 불필요
#             outputs = model(**inputs)
#             predictions = torch.softmax(outputs.logits, dim=-1)  # 확률값으로 변환
#             # 욕설 클래스([1])의 확률을 사용
#             profanity_prob = predictions[0][1].item()
        
#         return {
#             "is_profanity": profanity_prob > 0.5,
#             "confidence": profanity_prob
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/health")
# async def health_check():
#     """
#     서버 상태 확인용 엔드포인트
#     """
#     return {"status": "healthy"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

############################################################################################

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os
import boto3
import zipfile
from botocore.exceptions import NoCredentialsError, ClientError

app = FastAPI()

# CORS 설정 - 다른 도메인에서의 API 접근을 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 도메인만 허용하도록 수정 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델과 토크나이저를 전역 변수로 선언
model = None
tokenizer = None

def download_model_from_s3():
    """
    S3에서 모델 파일을 다운로드하는 함수
    
    Returns:
        bool: 다운로드 성공 여부
    """
    try:
        s3 = boto3.client('s3')
        # 기존 버킷 사용 시: leteatgo-s3-bucket/models/profanity_filter_model.zip
        # 새 버킷 사용 시: profanity-filter-model-bucket/profanity_filter_model.zip
        bucket_name = os.getenv('S3_BUCKET_NAME', 'leteatgo-photo-album')
        model_key = os.getenv('S3_MODEL_KEY', 'models/profanity_filter_model.zip')
        local_model_path = './profanity_filter_model'
        zip_path = './model.zip'
        
        # 모델 디렉토리가 이미 존재하는지 확인
        if os.path.exists(local_model_path) and os.listdir(local_model_path):
            print("Model already exists locally")
            return True
        
        print(f"Downloading model from S3: s3://{bucket_name}/{model_key}")
        
        # S3에서 압축된 모델 파일 다운로드
        s3.download_file(bucket_name, model_key, zip_path)
        
        # 압축 해제
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('./')
        
        # 임시 zip 파일 삭제
        os.remove(zip_path)
        
        print("Model downloaded and extracted successfully")
        return True
        
    except NoCredentialsError:
        print("AWS credentials not found")
        return False
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def initialize_model():
    """
    모델과 토크나이저를 초기화하는 함수
    
    Returns:
        bool: 초기화 성공 여부
    """
    global model, tokenizer
    
    try:
        model_path = './profanity_filter_model'
        
        # S3에서 모델 다운로드 시도 (로컬에 없는 경우)
        if not os.path.exists(model_path) or not os.listdir(model_path):
            if not download_model_from_s3():
                raise Exception("Failed to download model from S3")
        
        # 모델과 토크나이저 로드
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        
        print("Model and tokenizer loaded successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False

# 애플리케이션 시작 시 모델 초기화
@app.on_event("startup")
async def startup_event():
    if not initialize_model():
        raise Exception("Failed to initialize model")

# API 요청 데이터 모델 정의
class TextRequest(BaseModel):
    text: str  # 검사할 텍스트

@app.post("/predict")
async def predict(request: TextRequest):
    """
    텍스트의 욕설 여부를 검사하는 엔드포인트
    
    Args:
        request (TextRequest): 검사할 텍스트를 포함한 요청 객체
    
    Returns:
        dict: {
            "is_profanity": bool,  # 욕설 포함 여부
            "confidence": float    # 예측 신뢰도 (0.0 ~ 1.0)
        }
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # 텍스트를 모델이 이해할 수 있는 형태로 변환
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # 최대 길이 제한
        )
        
        # 모델을 통한 예측 수행
        with torch.no_grad():  # 추론 시에는 그래디언트 계산 불필요
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)  # 확률값으로 변환
            # 욕설 클래스([1])의 확률을 사용
            profanity_prob = predictions[0][1].item()
        
        return {
            "is_profanity": profanity_prob > 0.5,
            "confidence": profanity_prob
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    서버 상태 확인용 엔드포인트
    """
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


############################################################################################


# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
# import torch
# import os
# import boto3
# import zipfile
# from botocore.exceptions import NoCredentialsError, ClientError

# app = FastAPI()

# # CORS 설정 - 다른 도메인에서의 API 접근을 허용
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 실제 운영 환경에서는 특정 도메인만 허용하도록 수정 필요
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 모델과 토크나이저를 전역 변수로 선언
# model = None
# tokenizer = None

# def download_model_from_s3():
#     """
#     S3에서 모델 파일을 다운로드하는 함수
    
#     Returns:
#         bool: 다운로드 성공 여부
#     """
#     try:
#         s3 = boto3.client('s3')
#         bucket_name = os.getenv('S3_BUCKET_NAME', 'your-model-bucket')
#         model_key = os.getenv('S3_MODEL_KEY', 'profanity_filter_model.zip')
#         local_model_path = './profanity_filter_model'
#         zip_path = './model.zip'
        
#         # 모델 디렉토리가 이미 존재하는지 확인
#         if os.path.exists(local_model_path) and os.listdir(local_model_path):
#             print("Model already exists locally")
#             return True
        
#         print(f"Downloading model from S3: s3://{bucket_name}/{model_key}")
        
#         # S3에서 압축된 모델 파일 다운로드
#         s3.download_file(bucket_name, model_key, zip_path)
        
#         # 압축 해제
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall('./')
        
#         # 임시 zip 파일 삭제
#         os.remove(zip_path)
        
#         print("Model downloaded and extracted successfully")
#         return True
        
#     except NoCredentialsError:
#         print("AWS credentials not found")
#         return False
#     except ClientError as e:
#         print(f"Error downloading from S3: {e}")
#         return False
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#         return False

# def initialize_model():
#     """
#     모델과 토크나이저를 초기화하는 함수
    
#     Returns:
#         bool: 초기화 성공 여부
#     """
#     global model, tokenizer
    
#     try:
#         model_path = './profanity_filter_model'
        
#         # S3에서 모델 다운로드 시도 (로컬에 없는 경우)
#         if not os.path.exists(model_path) or not os.listdir(model_path):
#             if not download_model_from_s3():
#                 raise Exception("Failed to download model from S3")
        
#         # 모델과 토크나이저 로드
#         tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
#         model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        
#         print("Model and tokenizer loaded successfully")
#         return True
        
#     except Exception as e:
#         print(f"Error initializing model: {e}")
#         return False

# # 애플리케이션 시작 시 모델 초기화
# @app.on_event("startup")
# async def startup_event():
#     if not initialize_model():
#         raise Exception("Failed to initialize model")

# class TextInput(BaseModel):
#     text: str

# @app.post("/predict")
# async def predict(input_data: TextInput):
#     if model is None or tokenizer is None:
#         raise HTTPException(status_code=500, detail="Model not initialized")
    
#     try:
#         # 텍스트 토크나이징
#         inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
#         # 예측 수행
#         with torch.no_grad():
#             outputs = model(**inputs)
#             predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#             predicted_class = torch.argmax(predictions, dim=-1).item()
#             confidence = predictions[0][predicted_class].item()
        
#         # 결과 반환 (0: clean, 1: profanity)
#         is_profanity = predicted_class == 1
        
#         return {
#             "text": input_data.text,
#             "is_profanity": is_profanity,
#             "confidence": confidence,
#             "predicted_class": predicted_class
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "model_loaded": model is not None}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
