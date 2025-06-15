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


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os

app = FastAPI()

# CORS 설정 - 다른 도메인에서의 API 접근을 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 도메인만 허용하도록 수정 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# fine-tuned BERT 모델 경로 (너가 저장한 경로로 변경)
MODEL_PATH = "./profanity_filter_model"  # fine-tuned 모델 경로
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)  # 2개의 레이블: 욕설(1) / 비욕설(0)

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
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
