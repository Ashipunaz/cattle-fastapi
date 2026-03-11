from pydantic import BaseModel
from typing import List, Optional


class PredictionResult(BaseModel):
    id: str
    filename: str
    disease: str
    full_name: str
    confidence: float
    severity: str
    requires_vet: bool
    what_you_see: str
    what_to_do: str
    urgency_msg: str
    all_predictions: dict
    timestamp: str
    model_version: str


class HistoryItem(BaseModel):
    id: str
    filename: str
    disease: str
    full_name: str
    confidence: float
    severity: str
    timestamp: str
    model_version: str


class ModelVersion(BaseModel):
    version: str
    filename: str
    uploaded_at: str
    is_active: bool
    size_kb: float


class AdminTokenRequest(BaseModel):
    token: str


class ActivateModelRequest(BaseModel):
    version: str


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None