from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from backend.app.schemas import PredictionRequest, PredictionResponse
from backend.app.predictor import get_predictor, NewsPredictor
from backend.app.v2.router import router as v2_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load models on application startup
    get_predictor()
    yield

app = FastAPI(
    title="Fake News Detection & Real-Time Verification API",
    description="NLP and Machine Learning pipeline for fake news classification & V2 evidence verification",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for local React development and production simulation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register V2 verification endpoints
app.include_router(v2_router)
app.include_router(v2_router, prefix="/api")

@app.get("/", tags=["General"])
def read_root():
    return {
        "status": "ok",
        "service": "Fake News Detection API"
    }

@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "healthy"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_news(payload: PredictionRequest):
    try:
        predictor: NewsPredictor = get_predictor()
        result = predictor.predict(title=payload.title, text=payload.text)
        return PredictionResponse(**result)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction service error: {str(e)}"
        )
