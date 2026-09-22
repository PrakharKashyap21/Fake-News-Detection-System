from typing import Optional
from backend.app.predictor import get_predictor, NewsPredictor
from backend.app.v2.schemas import (
    LinguisticSignal,
    ExtractedClaim,
    ClaimVerificationResult
)


class SVMSignalProvider:
    """Wraps the V1 LinearSVC + TF-IDF model to provide stylistic/linguistic pattern scores."""

    def __init__(self):
        self._predictor: Optional[NewsPredictor] = None

    def _get_predictor(self) -> NewsPredictor:
        if self._predictor is None:
            self._predictor = get_predictor()
        return self._predictor

    def get_signal(self, title: str = "", text: str = "") -> LinguisticSignal:
        """Evaluates the linguistic pattern score using the V1 LinearSVC model."""
        predictor = self._get_predictor()
        res = predictor.predict(title=title, text=text)
        return LinguisticSignal(
            label=res["label"],
            prediction=res["prediction"],
            confidence=res["confidence"],
            message=res["message"]
        )

    def get_signal_for_claim(self, claim: ExtractedClaim) -> LinguisticSignal:
        """Evaluates the linguistic pattern score for an ExtractedClaim."""
        claim_text = claim.text if claim and claim.text else ""
        return self.get_signal(text=claim_text)


class SVMPipelineIntegrator:
    """Combines ExtractedClaim, SVMSignalProvider, and ClaimVerificationResult

    without allowing the SVM signal to alter or influence the claim verdict.
    """

    def __init__(self, svm_provider: Optional[SVMSignalProvider] = None):
        self.svm_provider = svm_provider or get_svm_signal_provider()

    def attach_signal_to_result(
        self,
        claim: ExtractedClaim,
        result: ClaimVerificationResult
    ) -> ClaimVerificationResult:
        """Attaches the independent V1 SVM linguistic signal to a ClaimVerificationResult.

        Guarantees that verdict, counts, conflict flag, and reasoning remain 100% un-modified.
        """
        signal = self.svm_provider.get_signal_for_claim(claim)

        return ClaimVerificationResult(
            claim_id=result.claim_id,
            verdict=result.verdict,  # Strictly un-modified
            supporting_evidence_count=result.supporting_evidence_count,  # Un-modified
            contradicting_evidence_count=result.contradicting_evidence_count,  # Un-modified
            neutral_evidence_count=result.neutral_evidence_count,  # Un-modified
            has_conflicting_evidence=result.has_conflicting_evidence,  # Un-modified
            reasoning=result.reasoning,  # Un-modified
            linguistic_signal=signal
        )


_svm_signal_instance: Optional[SVMSignalProvider] = None


def get_svm_signal_provider() -> SVMSignalProvider:
    global _svm_signal_instance
    if _svm_signal_instance is None:
        _svm_signal_instance = SVMSignalProvider()
    return _svm_signal_instance
