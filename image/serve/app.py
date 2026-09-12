"""
최소한의 모델 서빙 데모 (FastAPI) — CT Screening(v11).

전체 CI/CD·모니터링 파이프라인이 아니라, "학습된 모델을 실제로 서빙해본
경험"을 보여주는 최소 단위 데모다. RAG 챗봇(`llm/main.py`)과 같은 FastAPI
패턴을 재사용해서 포트폴리오 전체의 기술 스택 일관성을 유지했다.

입력 형식: `ct_train_preprocess_1.ipynb`의 Preprocessor가 만드는 것과 같은
포맷 — (3, 64, 224, 224) float16/float32 .npy 볼륨 (외상 특화 3채널 윈도잉:
Soft Tissue/Angio/Bowel). 원본 DICOM 시리즈를 직접 업로드받아 그 자리에서
전처리하는 것까지는 이 데모 범위 밖이다(전처리 자체가 무겁고, RSNA 원본
데이터 없이는 로컬에서 검증도 어려움) — 대신 "학습된 모델 가중치를 로드해서
실제로 추론 서버를 띄운다"는 핵심만 보여준다.

실행:
    pip install fastapi uvicorn python-multipart torch timm
    uvicorn app:app --reload --port 8001
    # http://localhost:8001 접속
"""
import io
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "monai_ct_convnext_v11_ep16.pth"
)

# Kaggle에서 ct_screening_inference.py + screening_eval.py로 950개 검증 케이스
# 실측(image/README.md "recall 우선 재평가" 참고): recall 90% 지점(오탐률
# 54.4%)을 채택. recall 99% 지점(threshold=0.1031)은 오탐률 88.5%로 거의
# 전원 플래그라 스크리닝 도구로서 의미가 없다고 판단해 기각.
SUSPICION_THRESHOLD = 0.2335


class ScreeningModel(nn.Module):
    """ct_screening_inference.py와 동일한 구조(v11, 이진 스크리닝 헤드)."""

    def __init__(self):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=False, num_classes=0, drop_path_rate=0.1
        )
        self.dim = self.backbone.num_features
        self.attention_net = nn.Sequential(
            nn.Linear(self.dim, 256), nn.Tanh(), nn.Dropout(0.3), nn.Linear(256, 1)
        )
        self.suspicion_head = nn.Sequential(
            nn.Linear(self.dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 2)
        )

    def forward(self, x):
        b, s, c, h, w = x.shape
        chunk_size = 8
        all_features = []
        for i in range(0, s, chunk_size):
            x_chunk = x[:, i:i + chunk_size].reshape(-1, c, h, w)
            feat_chunk = self.backbone(x_chunk).view(b, -1, self.dim)
            all_features.append(feat_chunk)
        features = torch.cat(all_features, dim=1)
        att_weights = F.softmax(self.attention_net(features), dim=1)
        combined = torch.sum(features * att_weights, dim=1)
        return self.suspicion_head(combined)


_model = None


def get_model():
    """가중치를 최초 요청 시 1회만 로드(지연 로딩)."""
    global _model
    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"모델 가중치를 찾을 수 없음: {MODEL_PATH}",
            )
        model = ScreeningModel().to(DEVICE)
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_ema", ckpt.get("model", ckpt))
        new_state_dict = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items()
        )
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        _model = model
    return _model


app = FastAPI(title="CT Screening Demo — 확진이 아니라 의심")


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!doctype html><html><head><meta charset="utf-8">
    <title>CT Screening 데모</title></head>
    <body style="font-family:sans-serif;max-width:640px;margin:40px auto;">
      <h2>CT Screening 데모 — 확진이 아니라 의심</h2>
      <p>전처리된 CT 볼륨(.npy, shape (3,64,224,224))을 업로드하면
      "정밀 검사가 필요할 수 있는가"만 판단합니다. 정확한 병명은 이 모델의
      역할이 아닙니다.</p>
      <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".npy" required>
        <button type="submit">추론 실행</button>
      </form>
    </body></html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail=".npy 파일만 지원합니다")

    raw = await file.read()
    try:
        volume = np.load(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f".npy 파싱 실패: {e}")

    if volume.shape != (3, 64, 224, 224):
        raise HTTPException(
            status_code=400,
            detail=f"입력 shape이 (3,64,224,224)이어야 함, 받은 shape: {volume.shape}",
        )

    model = get_model()
    x = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0)  # (1, 3, 64, 224, 224)
    # 모델은 (B, S, C, H, W) 슬라이스 시퀀스를 기대 — (3,64,224,224)를
    # (64, 3, 224, 224)로 바꿔서 배치 차원 추가
    x = x.permute(0, 2, 1, 3, 4).to(DEVICE)  # (1, 64, 3, 224, 224)

    with torch.no_grad():
        prob = F.softmax(model(x), dim=1)[0, 1].item()

    suspicion = prob >= SUSPICION_THRESHOLD
    return JSONResponse({
        "probability": round(prob, 4),
        "threshold": SUSPICION_THRESHOLD,
        "suspicion": suspicion,
        "message": (
            "이상 소견이 의심됩니다 — 정밀 검사를 권고합니다."
            if suspicion else
            "뚜렷한 이상 소견이 확인되지 않았습니다. (확진이 아닌 스크리닝 결과)"
        ),
    })
