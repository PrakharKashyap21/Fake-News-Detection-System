from fastapi import APIRouter, Depends, HTTPException, status
from backend.app.v2.schemas import VerificationRequest, VerificationResponse
from backend.app.v2.verification_service import get_verification_service, VerificationService

router = APIRouter(prefix="/v2", tags=["V2 Verification"])


@router.post("/verify", response_model=VerificationResponse)
def verify_news_v2(payload: VerificationRequest, service: VerificationService = Depends(get_verification_service)):
    try:
        return service.verify_news(payload)

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"V2 verification service error: {str(e)}"
        )
