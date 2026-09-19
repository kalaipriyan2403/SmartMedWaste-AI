from pydantic import BaseModel


class PredictionResponse(BaseModel):
    category: str
    colour_code: str
    confidence: float
    confidence_percentage: float
    requires_manual_review: bool