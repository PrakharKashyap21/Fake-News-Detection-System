from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class StanceType(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"


class EvidenceSourceType(str, Enum):
    FACT_CHECK_API = "FACT_CHECK_API"
    LIVE_NEWS_SEARCH = "LIVE_NEWS_SEARCH"


class ClaimVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"


class OverallAssessment(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"


class EvidenceItem(BaseModel):
    id: str
    claim_id: str
    source_type: EvidenceSourceType
    publisher: str
    domain: str
    url: str
    title: str
    snippet: str
    publish_date: Optional[str] = None
    credibility_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stance: StanceType
    raw_rating: Optional[str] = None


class ExtractedClaim(BaseModel):
    claim_id: str
    text: str
    verdict: ClaimVerdict = ClaimVerdict.UNVERIFIED
    support_score: float = 0.0
    contradict_score: float = 0.0
    keywords: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    explanation: str = "Insufficient external evidence available to independently verify this claim."


class LinguisticSignal(BaseModel):
    label: int
    prediction: str
    confidence: float
    message: str


class VerificationRequest(BaseModel):
    title: Optional[str] = ""
    text: Optional[str] = ""
    max_claims: Optional[int] = 5
    include_linguistic_signal: Optional[bool] = True

    @model_validator(mode="after")
    def validate_content_present(self) -> "VerificationRequest":
        title_str = (self.title or "").strip()
        text_str = (self.text or "").strip()
        if not title_str and not text_str:
            raise ValueError("At least one of 'title' or 'text' must be provided for verification.")
        return self


class VerificationResponse(BaseModel):
    overall_assessment: OverallAssessment
    assessment_summary: str
    confidence_score: float
    has_conflict: bool
    claims: List[ExtractedClaim]
    linguistic_signal: Optional[LinguisticSignal] = None
    disclaimer: str = (
        "Verification is based on aggregated live news and fact-check sources. "
        "UNVERIFIED claims do not imply falsity, but rather an absence of conclusive external reporting."
    )


class ClaimEvidenceSummary(BaseModel):
    claim_id: str
    claim_text: str
    total_evidence_count: int = 0
    fact_check_count: int = 0
    live_news_count: int = 0
    supporting_evidence_count: int = 0
    contradicting_evidence_count: int = 0
    neutral_evidence_count: int = 0
    unique_source_domains: List[str] = Field(default_factory=list)
    unique_domain_count: int = 0
    most_recent_evidence_timestamp: Optional[str] = None
    has_conflicting_evidence: bool = False
    fact_check_evidence: List[EvidenceItem] = Field(default_factory=list)
    live_news_evidence: List[EvidenceItem] = Field(default_factory=list)
    all_evidence: List[EvidenceItem] = Field(default_factory=list)


class EvidenceStrength(str, Enum):
    NONE = "NONE"
    LIMITED = "LIMITED"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class UncertaintyLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ClaimVerificationResult(BaseModel):
    claim_id: str
    verdict: ClaimVerdict
    supporting_evidence_count: int = 0
    contradicting_evidence_count: int = 0
    neutral_evidence_count: int = 0
    has_conflicting_evidence: bool = False
    reasoning: str
    evidence_strength: EvidenceStrength = EvidenceStrength.NONE
    uncertainty_level: UncertaintyLevel = UncertaintyLevel.HIGH
    linguistic_signal: Optional[LinguisticSignal] = None




