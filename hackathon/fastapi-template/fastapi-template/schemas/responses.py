# schemas/responses.py
from pydantic import BaseModel
from typing import Optional

class BaseResponse(BaseModel):
    status: str = "success"
    message: str

class PredictionResponse(BaseResponse):
    filename: Optional[str] = None
    total_rows: Optional[int] = None
    predictions_positives: Optional[int] = None
    predictions_negatives: Optional[int] = None

class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str
