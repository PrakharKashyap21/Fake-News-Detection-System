from typing import List, Optional, Tuple
from backend.app.v2.schemas import (
    ExtractedClaim,
    EvidenceItem,
    ClaimEvidenceSummary,
    ClaimVerdict,
    ClaimVerificationResult,
    EvidenceStrength,
    UncertaintyLevel,
    StanceType
)
from backend.app.v2.evidence_aggregator import get_evidence_aggregator, EvidenceAggregator


def evaluate_uncertainty(summary: ClaimEvidenceSummary) -> Tuple[EvidenceStrength, UncertaintyLevel]:
    """Evaluates evidence strength and uncertainty level based strictly on the aggregated evidence summary."""
    if not summary or summary.total_evidence_count == 0:
        return EvidenceStrength.NONE, UncertaintyLevel.HIGH

    supp_count = summary.supporting_evidence_count
    contra_count = summary.contradicting_evidence_count
    has_conflict = summary.has_conflicting_evidence

    # If evidence items exist, but no explicit supporting or contradicting fact-check stance exists
    if supp_count == 0 and contra_count == 0:
        return EvidenceStrength.LIMITED, UncertaintyLevel.HIGH

    # If supporting and contradicting evidence conflict
    if has_conflict:
        return EvidenceStrength.LIMITED, UncertaintyLevel.HIGH

    # Count unique independent fact-check source domains with explicit stances
    explicit_domains = set()
    if summary.fact_check_evidence:
        for item in summary.fact_check_evidence:
            if item.stance in (StanceType.SUPPORTS, StanceType.CONTRADICTS):
                clean_dom = str(item.domain).strip().lower() if item.domain else ""
                if clean_dom and clean_dom != "unknown":
                    explicit_domains.add(clean_dom)

    unique_explicit_domains = len(explicit_domains)

    if unique_explicit_domains >= 2:
        return EvidenceStrength.STRONG, UncertaintyLevel.LOW

    return EvidenceStrength.MODERATE, UncertaintyLevel.MEDIUM


class VerdictEngine:
    """Deterministic, transparent claim verification engine with explicit evidence strength and uncertainty analysis."""

    def __init__(self, aggregator: Optional[EvidenceAggregator] = None):
        self.aggregator = aggregator or get_evidence_aggregator()

    def verify_summary(self, summary: ClaimEvidenceSummary) -> ClaimVerificationResult:
        """Determines the claim verdict, evidence strength, uncertainty level, and reasoning from summary."""
        claim_id = summary.claim_id if summary and summary.claim_id else "unknown_claim"
        supp_count = summary.supporting_evidence_count if summary else 0
        contra_count = summary.contradicting_evidence_count if summary else 0
        neut_count = summary.neutral_evidence_count if summary else 0

        strength, uncertainty = evaluate_uncertainty(summary)

        # Rule 4: Both supporting and contradicting evidence exist
        if supp_count > 0 and contra_count > 0:
            return ClaimVerificationResult(
                claim_id=claim_id,
                verdict=ClaimVerdict.UNVERIFIED,
                supporting_evidence_count=supp_count,
                contradicting_evidence_count=contra_count,
                neutral_evidence_count=neut_count,
                has_conflicting_evidence=True,
                reasoning="Supporting and contradicting evidence were both found; the claim is unverified.",
                evidence_strength=strength,
                uncertainty_level=uncertainty
            )

        # Rule 2: Supporting fact-check evidence exists, no contradicting evidence
        if supp_count > 0 and contra_count == 0:
            return ClaimVerificationResult(
                claim_id=claim_id,
                verdict=ClaimVerdict.SUPPORTED,
                supporting_evidence_count=supp_count,
                contradicting_evidence_count=contra_count,
                neutral_evidence_count=neut_count,
                has_conflicting_evidence=False,
                reasoning="Supporting fact-check evidence was found with no contradicting evidence.",
                evidence_strength=strength,
                uncertainty_level=uncertainty
            )

        # Rule 3: Contradicting fact-check evidence exists, no supporting evidence
        if contra_count > 0 and supp_count == 0:
            return ClaimVerificationResult(
                claim_id=claim_id,
                verdict=ClaimVerdict.CONTRADICTED,
                supporting_evidence_count=supp_count,
                contradicting_evidence_count=contra_count,
                neutral_evidence_count=neut_count,
                has_conflicting_evidence=False,
                reasoning="Contradicting fact-check evidence was found with no supporting evidence.",
                evidence_strength=strength,
                uncertainty_level=uncertainty
            )

        # Rule 1, 5, 6: No supporting or contradicting evidence found
        return ClaimVerificationResult(
            claim_id=claim_id,
            verdict=ClaimVerdict.UNVERIFIED,
            supporting_evidence_count=supp_count,
            contradicting_evidence_count=contra_count,
            neutral_evidence_count=neut_count,
            has_conflicting_evidence=False,
            reasoning="No supporting or contradicting evidence was found.",
            evidence_strength=strength,
            uncertainty_level=uncertainty
        )

    def verify_claim(
        self,
        claim: ExtractedClaim,
        evidence_items: Optional[List[EvidenceItem]] = None
    ) -> ClaimVerificationResult:
        """Aggregates evidence for an ExtractedClaim and evaluates its verdict."""
        summary = self.aggregator.aggregate_evidence(claim, evidence_items)
        return self.verify_summary(summary)


_verdict_engine_instance = None


def get_verdict_engine() -> VerdictEngine:
    global _verdict_engine_instance
    if _verdict_engine_instance is None:
        _verdict_engine_instance = VerdictEngine()
    return _verdict_engine_instance
