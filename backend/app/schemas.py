from typing import Optional
from pydantic import BaseModel, model_validator

class PredictionRequest(BaseModel):
    title: Optional[str] = ""
    text: Optional[str] = ""

    @model_validator(mode="after")
    def check_at_least_one_field_non_empty(self) -> "PredictionRequest":
        title_str = (self.title or "").strip()
        text_str = (self.text or "").strip()
        if not title_str and not text_str:
            raise ValueError("At least one of 'title' or 'text' must contain non-whitespace content.")
        return self

class PredictionResponse(BaseModel):
    prediction: str
    label: int
    confidence: float
    message: str
