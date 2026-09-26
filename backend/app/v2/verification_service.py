from typing import List, Optional, Dict, Any
from backend.app.v2.schemas import (
    VerificationRequest,
    VerificationResponse,
    ClaimVerificationDetail,
    OverallAssessment,
    ClaimVerdict,
    ExtractedClaim
)
from backend.app.v2.claim_extractor import get_claim_extractor, ClaimExtractor
from backend.app.v2.fact_check_retriever import (
    get_fact_check_retriever,
    FactCheckAPIKeyError,
    FactCheckAPIError,
    GoogleFactCheckRetriever
)
from backend.app.v2.newsapi_retriever import (
    get_newsapi_retriever,
    NewsAPIRetriever,
    NewsAPIKeyError,
    NewsAPIError,
    NewsAPIRateLimitError
)
from backend.app.v2.news_retriever import (
    NewsRetrieverAPIError,
    NewsRetrieverRateLimitError,
    GDELTNewsRetriever
)
from backend.app.v2.evidence_aggregator import get_evidence_aggregator, EvidenceAggregator
from backend.app.v2.evidence_matcher import get_evidence_matcher, EvidenceMatcher
from backend.app.v2.stance_analyzer import get_stance_analyzer, EvidenceStanceAnalyzer
from backend.app.v2.verdict_engine import get_verdict_engine, VerdictEngine
from backend.app.v2.svm_signal import get_svm_signal_provider, SVMSignalProvider, SVMPipelineIntegrator


class VerificationService:
    """Orchestrates V2 verification workflow combining claim extraction, fact-check retrieval,

    live news retrieval (NewsAPI), evidence relevance matching, evidence stance analysis, evidence aggregation, verdict evaluation, and SVM signals.
    """

    def __init__(
        self,
        claim_extractor: Optional[ClaimExtractor] = None,
        fc_retriever: Optional[GoogleFactCheckRetriever] = None,
        news_retriever: Optional[Any] = None,
        evidence_matcher: Optional[EvidenceMatcher] = None,
        stance_analyzer: Optional[EvidenceStanceAnalyzer] = None,
        aggregator: Optional[EvidenceAggregator] = None,
        verdict_engine: Optional[VerdictEngine] = None,
        svm_provider: Optional[SVMSignalProvider] = None,
        mock_mode: bool = False
    ):
        self.claim_extractor = claim_extractor or get_claim_extractor()
        self.fc_retriever = fc_retriever or get_fact_check_retriever(mock_mode=mock_mode)
        self.news_retriever = news_retriever or get_newsapi_retriever(mock_mode=mock_mode)
        self.evidence_matcher = evidence_matcher or get_evidence_matcher()
        self.stance_analyzer = stance_analyzer or get_stance_analyzer()
        self.aggregator = aggregator or get_evidence_aggregator()
        self.verdict_engine = verdict_engine or get_verdict_engine()
        self.svm_provider = svm_provider or get_svm_signal_provider()
        self.integrator = SVMPipelineIntegrator(self.svm_provider)


    def verify_news(self, request: VerificationRequest) -> VerificationResponse:
        title = (request.title or "").strip()
        text = (request.text or "").strip()
        max_claims = request.max_claims or 5

        # 1. Claim extraction
        extracted_claims = self.claim_extractor.extract_claims(title=title, text=text, max_claims=max_claims)

        # 2. Article-level V1 SVM Linguistic Signal
        article_svm_signal = None
        if request.include_linguistic_signal:
            try:
                article_svm_signal = self.svm_provider.get_signal(title=title, text=text)
            except Exception:
                pass

        service_status: Dict[str, str] = {
            "fact_check_api": "ok",
            "live_news_api": "ok"
        }

        claim_details: List[ClaimVerificationDetail] = []

        for claim in extracted_claims:
            # Fact-check evidence retrieval
            fc_evidence = []
            try:
                fc_evidence = self.fc_retriever.search_claim(claim)
            except FactCheckAPIKeyError:
                service_status["fact_check_api"] = "missing_api_key"
            except FactCheckAPIError as fc_err:
                service_status["fact_check_api"] = f"error_{fc_err.status_code}"
            except Exception:
                service_status["fact_check_api"] = "error"

            # Live-news evidence retrieval (NewsAPI active provider)
            news_evidence = []
            try:
                news_evidence = self.news_retriever.search_claim_news(claim)
            except (NewsAPIKeyError, FactCheckAPIKeyError):
                service_status["live_news_api"] = "missing_api_key"
            except (NewsAPIRateLimitError, NewsRetrieverRateLimitError):
                service_status["live_news_api"] = "rate_limited"
            except (NewsAPIError, NewsRetrieverAPIError) as news_err:
                if news_err.status_code == 429:
                    service_status["live_news_api"] = "rate_limited"
                elif news_err.status_code == 401:
                    service_status["live_news_api"] = "error_401"
                elif news_err.status_code == 504:
                    service_status["live_news_api"] = "error_504"
                else:
                    service_status["live_news_api"] = f"error_{news_err.status_code}"
            except Exception:
                service_status["live_news_api"] = "error"

            combined_evidence = fc_evidence + news_evidence

            # Evidence relevance matching
            matched_evidence = self.evidence_matcher.process_claim_evidence(claim, combined_evidence)

            # Evidence stance analysis
            analyzed_evidence = self.stance_analyzer.process_claim_evidence_stance(claim, matched_evidence)

            # Evidence aggregation
            summary = self.aggregator.aggregate_evidence(claim, analyzed_evidence)


            # Claim verdict evaluation
            base_result = self.verdict_engine.verify_summary(summary)

            # Attach claim-level SVM signal without mutating verdict
            res_with_svm = self.integrator.attach_signal_to_result(claim, base_result)

            claim_details.append(
                ClaimVerificationDetail(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    verdict=res_with_svm.verdict,
                    reasoning=res_with_svm.reasoning,
                    evidence_strength=res_with_svm.evidence_strength,
                    uncertainty_level=res_with_svm.uncertainty_level,
                    has_conflicting_evidence=res_with_svm.has_conflicting_evidence,
                    supporting_evidence_count=res_with_svm.supporting_evidence_count,
                    contradicting_evidence_count=res_with_svm.contradicting_evidence_count,
                    neutral_evidence_count=res_with_svm.neutral_evidence_count,
                    keywords=claim.keywords,
                    evidence_summary=summary,
                    evidence=summary.all_evidence,
                    linguistic_signal=res_with_svm.linguistic_signal
                )
            )

        # Synthesize overall assessment deterministically and conservatively
        has_supported = any(c.verdict == ClaimVerdict.SUPPORTED for c in claim_details)
        has_contradicted = any(c.verdict == ClaimVerdict.CONTRADICTED for c in claim_details)
        has_conflict = any(c.has_conflicting_evidence for c in claim_details) or (has_supported and has_contradicted)

        if not claim_details:
            overall = OverallAssessment.UNVERIFIED
            summary_msg = "No factual claims were extracted from the input text."
        elif has_supported and has_contradicted:
            overall = OverallAssessment.UNVERIFIED
            summary_msg = "Mixed claim results: article contains both supported and contradicted claims."
        elif has_supported and all(c.verdict == ClaimVerdict.SUPPORTED for c in claim_details):
            overall = OverallAssessment.SUPPORTED
            summary_msg = "All claims in this article were confirmed by factual evidence."
        elif has_contradicted and all(c.verdict in (ClaimVerdict.CONTRADICTED, ClaimVerdict.UNVERIFIED) for c in claim_details):
            if all(c.verdict == ClaimVerdict.CONTRADICTED for c in claim_details):
                overall = OverallAssessment.CONTRADICTED
                summary_msg = "All claims in this article were contradicted by factual evidence."
            else:
                overall = OverallAssessment.CONTRADICTED
                summary_msg = "One or more claims in this article were contradicted by factual evidence, with no supported claims."
        elif has_supported and not has_contradicted:
            overall = OverallAssessment.UNVERIFIED
            summary_msg = "Some claims in this article were supported, but others remain unverified."
        else:
            overall = OverallAssessment.UNVERIFIED
            summary_msg = "Insufficient conclusive evidence found to verify the claims in this article."


        return VerificationResponse(
            overall_assessment=overall,
            assessment_summary=summary_msg,
            has_conflict=has_conflict,
            claims=claim_details,
            linguistic_signal=article_svm_signal,
            service_status=service_status
        )


_verification_service_instance = None


def get_verification_service(mock_mode: bool = False) -> VerificationService:
    global _verification_service_instance
    if _verification_service_instance is None or mock_mode:
        _verification_service_instance = VerificationService(mock_mode=mock_mode)
    return _verification_service_instance
