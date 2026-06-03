from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from backend.inference import predict_from_landmarks, predict_from_image

class PredictRequest(BaseModel):
    landmarks: Optional[List[List[float]]] = None

router = APIRouter()

@router.post("/predict")
async def predict(req: PredictRequest):
    try:
        if req.landmarks:
            pred, conf = predict_from_landmarks(req.landmarks)
            return {"prediction": pred, "confidence": conf}
        raise HTTPException(status_code=400, detail="Missing landmarks")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract")
async def extract_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pred, conf, bbox = predict_from_image(contents)
        return {"prediction": pred, "confidence": conf, "bbox": bbox}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    return {"status": "ok"}
